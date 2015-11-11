#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
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
#include "cuda/common.h"
#include "cuda/init_cuda.h"
#include "cuda/me.h"

extern "C" {
#include "tables.h"
#include "simd/common.h"
#include "simd/me.h"
}


static const int Y = Y_COMPONENT;
static const int U = U_COMPONENT;
static const int V = V_COMPONENT;

/* getopt */
extern int optind;
extern char *optarg;

static void c63_encode_image(struct c63_common *cm, const struct c63_common_gpu& cm_gpu,
		const struct c63_cuda& c63_cuda, struct segment_yuv* image_gpu)
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

	cudaMemcpy(cm->curframe->orig->Y, (void*) cm->curframe->orig_gpu->Y, cm->ypw * cm->yph * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(cm->curframe->orig->U, (void*) cm->curframe->orig_gpu->U, cm->upw * cm->uph * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(cm->curframe->orig->V, (void*) cm->curframe->orig_gpu->V, cm->vpw * cm->vph * sizeof(uint8_t), cudaMemcpyDeviceToHost);

	if (!cm->curframe->keyframe)
	{
		/* Motion Estimation */
		c63_motion_estimate_gpu(cm, cm_gpu, c63_cuda);
		c63_motion_estimate_host(cm);

		/* Motion Compensation */
		c63_motion_compensate_gpu(cm, c63_cuda);
		c63_motion_compensate_host(cm);
	}
	else
	{
		// dct_quantize() expects zeroed out prediction buffers for key frames.
		// We zero them out here since we reuse the buffers from previous frames.
		zero_out_prediction_gpu(cm, c63_cuda);
		zero_out_prediction_host(cm);
	}

	/* DCT and Quantization */
	dct_quantize_gpu(cm, c63_cuda);
	dct_quantize_host(cm);

	/* Reconstruct frame for inter-prediction */
	dequantize_idct_gpu(cm, c63_cuda);
	dequantize_idct_host(cm);

	/* Function dump_image(), found in common.c, can be used here to check if the
	 prediction is correct */
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

void interrupt_handler(int signal)
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

	sigaction(SIGINT, &int_handler, NULL);

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

	struct c63_cuda c63_cuda = init_c63_cuda();
	struct c63_common *cm = init_c63_common(width, height, c63_cuda);
	struct c63_common_gpu cm_gpu = init_c63_gpu(cm, c63_cuda);

	set_sizes_offsets(cm);

	struct segment_yuv images_gpu[2];
	images_gpu[0] = init_image_segment(cm, 0);
	init_remote_encoded_data_segment(0);
	init_remote_encoded_data_segment(1);
	init_local_encoded_data_segment();

	//yuv_t* image_gpu = create_image_gpu(cm);
	int segNum = 0;

	int transferred = 0;
	while (1)
	{
		// The reader sends an interrupt when it has transferred the next frame
		int done = wait_for_reader();

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

			wait_for_writer();

			// Send interrupt to writer signaling that encoding has been finished
			signal_writer(ENCODING_FINISHED);
			break;
		}

		c63_encode_image(cm, cm_gpu, c63_cuda, &images_gpu[0]);

		// Wait until the GPU has finished encoding
		cudaStreamSynchronize(c63_cuda.stream[Y]);
		cudaStreamSynchronize(c63_cuda.stream[U]);
		cudaStreamSynchronize(c63_cuda.stream[V]);

		printf(", encoded\n");
		fflush(stdout);

		if (numframes != 0 && transferred == 2)
		{
			// The writer sends an interrupt when it is ready for the next frame
			wait_for_writer();
			--transferred;
		}

		// Copy data frame to remote segment - interrupt to writer handled by callback
		transfer_encoded_data(cm->curframe->keyframe, cm->curframe->mbs, cm->curframe->residuals,
				segNum);
		++transferred;

		// Reader can transfer next frame
		signal_reader();

		// Send interrupt to writer signaling the data has been transfered
		//signal_writer(DATA_TRANSFERRED);

		++cm->framenum;
		++cm->frames_since_keyframe;

		++numframes;

		segNum ^= 1;
	}

	//destroy_image_gpu(image_gpu);

	cleanup_c63_gpu(cm_gpu);
	cleanup_c63_common(cm);
	cleanup_c63_cuda(c63_cuda);

	cleanup_segments();
	cleanup_SISCI();

	return EXIT_SUCCESS;
}
