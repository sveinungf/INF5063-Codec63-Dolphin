#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <sisci_api.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <utility>

#include "allocation.h"
#include "c63.h"
#include "init.h"
#include "sisci.h"
#include "cuda/init_cuda.h"

extern "C" {
#include "simd/me.h"
}


static const int on_gpu[COLOR_COMPONENTS] = { Y_ON_GPU, U_ON_GPU, V_ON_GPU };

static const int Y = Y_COMPONENT;
static const int U = U_COMPONENT;
static const int V = V_COMPONENT;

/* getopt */
extern int optind;
extern char *optarg;

static struct c63_common* cm;
static struct c63_cuda c63_cuda;

static pthread_mutex_t mutex_parent[COLOR_COMPONENTS];
static pthread_mutex_t mutex_simd[COLOR_COMPONENTS];
static pthread_cond_t cond_frame_received[COLOR_COMPONENTS];
static pthread_cond_t cond_frame_encoded[COLOR_COMPONENTS];
static bool thread_done = false;

template<int component>
static inline void c63_encode_image_host()
{
	uint8_t* orig;
	const volatile uint8_t* orig_gpu;

	switch (component) {
		case Y_COMPONENT:
			orig = cm->curframe->orig->Y;
			orig_gpu = cm->curframe->orig_gpu->Y;
			break;
		case U_COMPONENT:
			orig = cm->curframe->orig->U;
			orig_gpu = cm->curframe->orig_gpu->U;
			break;
		case V_COMPONENT:
			orig = cm->curframe->orig->V;
			orig_gpu = cm->curframe->orig_gpu->V;
			break;
	}

	const int w = cm->padw[component];
	const int h = cm->padh[component];
	const size_t size = w * h * sizeof(uint8_t);
	const cudaStream_t stream = c63_cuda.memcpy_stream[component];

	// Using async memcpy so we don't have to wait for the other streams to finish
	cudaMemcpyAsync(orig, (void*) orig_gpu, size, cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	if (!cm->curframe->keyframe)
	{
		/* Motion Estimation */
		c63_motion_estimate(cm, component);

		/* Motion Compensation */
		c63_motion_compensate(cm, component);
	}
	else
	{
		// dct_quantize() expects zeroed out prediction buffers for key frames.
		// We zero them out here since we reuse the buffers from previous frames.
		zero_out_prediction_host<component>(cm);
	}

	/* DCT and Quantization */
	dct_quantize_host<component>(cm);

	/* Reconstruct frame for inter-prediction */
	dequantize_idct_host<component>(cm);

	/* Function dump_image(), found in common.c, can be used here to check if the
	 prediction is correct */
}

template<int component>
static void* thread_c63_encode_image_host(void*)
{
	pthread_mutex_lock(&mutex_parent[component]);
	while (!thread_done) {
		pthread_cond_wait(&cond_frame_received[component], &mutex_parent[component]);

		if (thread_done) {
			break;
		}

		c63_encode_image_host<component>();

		pthread_mutex_lock(&mutex_simd[component]);
		pthread_cond_signal(&cond_frame_encoded[component]);
		pthread_mutex_unlock(&mutex_simd[component]);
	}

	pthread_mutex_unlock(&mutex_parent[component]);

	return nullptr;
}

static inline void c63_encode_image_gpu(const struct c63_common_gpu& cm_gpu)
{
	if (!cm->curframe->keyframe)
	{
		c63_motion_estimate_gpu(cm, cm_gpu, c63_cuda);
		c63_motion_compensate_gpu(cm, c63_cuda);
	}
	else
	{
		zero_out_prediction_gpu(cm, c63_cuda);
	}

	dct_quantize_gpu(cm, c63_cuda);
	dequantize_idct_gpu(cm, c63_cuda);
}

static void c63_encode_image(const struct c63_common_gpu& cm_gpu, struct segment_yuv* image_gpu)
{
	// Advance to next frame by swapping current and reference frame
	std::swap(cm->curframe, cm->refframe);

	cm->curframe->orig_gpu = image_gpu;

	/* Check if keyframe */
	if (cm->framenum == 0 || cm->frames_since_keyframe == cm->keyframe_interval)
	{
		cm->curframe->keyframe = 1;
		cm->frames_since_keyframe = 0;
	}
	else
	{
		cm->curframe->keyframe = 0;
	}

	for (int c = 0; c < COLOR_COMPONENTS; ++c) {
		if (!on_gpu[c]) {
			pthread_mutex_lock(&mutex_parent[c]);
			pthread_cond_signal(&cond_frame_received[c]);
			pthread_mutex_unlock(&mutex_parent[c]);
		}
	}

	c63_encode_image_gpu(cm_gpu);
	//c63_encode_image_host();


}

static void print_help()
{
	printf("Usage: ./c63enc [options]\n");
	printf("Command line options:\n");
	printf("  -a                             Local adapter number\n");
	printf("  -r                             Reader node ID\n");
	printf("  -w							 Writer node ID\n");
	printf("\n");

	exit(EXIT_FAILURE);
}

void interrupt_handler(int)
{
	SCITerminate();
	exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
	struct sigaction int_handler;
	int_handler.sa_handler = interrupt_handler;
	sigemptyset(&int_handler.sa_mask);
	int_handler.sa_flags = 0;

	sigaction(SIGINT, &int_handler, nullptr);

	int c;

	if (argc == 1)
	{
		print_help();
	}

	unsigned int localAdapterNo = 0;
	unsigned int readerNodeId = 0;
	unsigned int writerNodeId = 0;

	while ((c = getopt(argc, argv, "a:r:w:")) != -1)
	{
		switch (c)
		{
			case 'a':
				localAdapterNo = atoi(optarg);
				break;
			case 'r':
				readerNodeId = atoi(optarg);
				break;
			case 'w':
				writerNodeId = atoi(optarg);
				break;
			default:
				print_help();
				break;
		}
	}

	if (optind > argc)
	{
		fprintf(stderr, "Error getting program options, try --help.\n");
		exit(EXIT_FAILURE);
	}

	/* Encode input frames */
	int numframes = 0;

	init_SISCI(localAdapterNo, readerNodeId, writerNodeId);

	uint32_t width, height;
	receive_width_and_height(&width, &height);
	send_width_and_height(width, height);

	c63_cuda = init_c63_cuda();
	cm = init_c63_common(width, height, c63_cuda);
	struct c63_common_gpu cm_gpu = init_c63_gpu(cm, c63_cuda);

	set_sizes_offsets(cm);

	struct segment_yuv images_gpu[2];
	images_gpu[0] = init_image_segment(cm, 0);
	images_gpu[1] = init_image_segment(cm, 1);
	init_remote_encoded_data_segment(0);
	init_remote_encoded_data_segment(1);
	init_local_encoded_data_segments();

	//yuv_t* image_gpu = create_image_gpu(cm);
	int segNum = 0;

	pthread_t simd_threads[COLOR_COMPONENTS];

	for (int c = 0; c < COLOR_COMPONENTS; ++c) {
		if (!on_gpu[c]) {
			mutex_parent[c] = PTHREAD_MUTEX_INITIALIZER;
			mutex_simd[c] = PTHREAD_MUTEX_INITIALIZER;
			cond_frame_received[c] = PTHREAD_COND_INITIALIZER;
			cond_frame_encoded[c] = PTHREAD_COND_INITIALIZER;
		}
	}

#if !(Y_ON_GPU)
	pthread_create(&simd_threads[Y], nullptr, thread_c63_encode_image_host<Y>, nullptr);
#endif
#if !(U_ON_GPU)
	pthread_create(&simd_threads[U], nullptr, thread_c63_encode_image_host<U>, nullptr);
#endif
#if !(V_ON_GPU)
	pthread_create(&simd_threads[V], nullptr, thread_c63_encode_image_host<V>, nullptr);
#endif

	for (int c = 0; c < COLOR_COMPONENTS; ++c) {
		if (!on_gpu[c]) {
			pthread_mutex_lock(&mutex_simd[c]);
		}
	}

	int transferred = 0;
	while (1)
	{
		// The reader sends an interrupt when it has transferred the next frame
		int done = wait_for_reader(segNum);

		printf("Frame %d:", numframes);
		fflush(stdout);

		if (!done)
		{
			printf(" Received");
			fflush(stdout);
		}
		else
		{
			printf("\rNo more frames from reader\n");

			wait_for_writer(segNum^1);

			// Send interrupt to writer signaling that encoding has been finished
			signal_writer(ENCODING_FINISHED, segNum);
			break;
		}


		c63_encode_image(cm_gpu, &images_gpu[segNum]);



#if Y_ON_GPU
		// Wait until the GPU has finished encoding
		cudaStreamSynchronize(c63_cuda.memcpy_stream[Y]);
#else
		// Or wait until the SIMD thread has finished encoding
		pthread_cond_wait(&cond_frame_encoded[Y], &mutex_simd[Y]);
#endif
#if U_ON_GPU
		cudaStreamSynchronize(c63_cuda.memcpy_stream[U]);
#else
		pthread_cond_wait(&cond_frame_encoded[U], &mutex_simd[U]);
#endif
#if V_ON_GPU
		cudaStreamSynchronize(c63_cuda.memcpy_stream[V]);
#else
		pthread_cond_wait(&cond_frame_encoded[V], &mutex_simd[V]);
#endif

		// Reader can transfer next frame
		signal_reader(segNum);

		printf(", encoded\n");
		fflush(stdout);

		wait_for_image_transfer(segNum);

		copy_to_segment(cm->curframe->mbs, cm->curframe->residuals, segNum);
		//cuda_copy_to_segment(cm, segNum);

		if (numframes >= NUM_IMAGE_SEGMENTS) {
			// The writer sends an interrupt when it is ready for the next frame
			wait_for_writer(segNum);
			//copy_to_segment(cm->curframe->mbs, cm->curframe->residuals, segNum);
			--transferred;
		}

		// Copy data frame to remote segment - interrupt to writer handled by callback

		transfer_encoded_data(cm->curframe->keyframe, segNum);
		++transferred;

		++cm->framenum;
		++cm->frames_since_keyframe;

		++numframes;

		segNum ^= 1;
	}

	for (int c = 0; c < COLOR_COMPONENTS; ++c) {
		if (!on_gpu[c]) {
			pthread_mutex_unlock(&mutex_simd[c]);
			pthread_mutex_lock(&mutex_parent[c]);
		}
	}

	thread_done = true;

	for (int c = 0; c < COLOR_COMPONENTS; ++c) {
		if (!on_gpu[c]) {
			pthread_cond_signal(&cond_frame_received[c]);
			pthread_mutex_unlock(&mutex_parent[c]);
		}
	}

	cleanup_c63_gpu(cm_gpu);
	cleanup_c63_common(cm);
	cleanup_c63_cuda(c63_cuda);

	cleanup_segments();
	cleanup_SISCI();

	cudaDeviceReset();

	for (int c = 0; c < COLOR_COMPONENTS; ++c) {
		if (!on_gpu[c]) {
			pthread_cond_destroy(&cond_frame_received[c]);
			pthread_cond_destroy(&cond_frame_encoded[c]);
			pthread_mutex_destroy(&mutex_parent[c]);
			pthread_mutex_destroy(&mutex_simd[c]);
		}
	}

	return EXIT_SUCCESS;
}
