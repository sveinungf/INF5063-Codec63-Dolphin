#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
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

struct c63_common* init_c63_enc(int width, int height, const struct c63_cuda& c63_cuda)
{
	/* calloc() sets allocated memory to zero */
	struct c63_common *cm = (struct c63_common*) calloc(1, sizeof(struct c63_common));

	cm->width = width;
	cm->height = height;

	cm->padw[Y] = cm->ypw = (uint32_t) (ceil(width / 16.0f) * 16);
	cm->padh[Y] = cm->yph = (uint32_t) (ceil(height / 16.0f) * 16);
	cm->padw[U] = cm->upw = (uint32_t) (ceil(width * UX / (YX * 8.0f)) * 8);
	cm->padh[U] = cm->uph = (uint32_t) (ceil(height * UY / (YY * 8.0f)) * 8);
	cm->padw[V] = cm->vpw = (uint32_t) (ceil(width * VX / (YX * 8.0f)) * 8);
	cm->padh[V] = cm->vph = (uint32_t) (ceil(height * VY / (YY * 8.0f)) * 8);

	cm->mb_cols[Y] = cm->ypw / 8;
	cm->mb_cols[U] = cm->mb_cols[Y] / 2;
	cm->mb_cols[V] = cm->mb_cols[U];

	cm->mb_rows[Y] = cm->yph / 8;
	cm->mb_rows[U] = cm->mb_rows[Y] / 2;
	cm->mb_rows[V] = cm->mb_rows[U];

	/* Quality parameters -- Home exam deliveries should have original values,
	 i.e., quantization factor should be 25, search range should be 16, and the
	 keyframe interval should be 100. */
	cm->qp = 25;                  // Constant quantization factor. Range: [1..50]
	//cm->me_search_range = 16;   // This is now defined in c63.h
	cm->keyframe_interval = 100;  // Distance between keyframes

	/* Initialize quantization tables */
	for (int i = 0; i < 64; ++i)
	{
		cm->quanttbl[Y][i] = yquanttbl_def[i] / (cm->qp / 10.0);
		cm->quanttbl[U][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
		cm->quanttbl[V][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
	}

	init_me_boundaries(cm);

	cm->curframe = create_frame(cm, c63_cuda);
	cm->refframe = create_frame(cm, c63_cuda);

	return cm;
}

void free_c63_enc(struct c63_common* cm)
{
	cleanup_me_boundaries(cm);

	destroy_frame(cm->curframe);
	destroy_frame(cm->refframe);

	free(cm);
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

int main(int argc, char **argv)
{
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
	struct c63_common *cm = init_c63_enc(width, height, c63_cuda);
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
	free_c63_enc(cm);
	cleanup_c63_cuda(c63_cuda);

	cleanup_segments();
	cleanup_SISCI();

	return EXIT_SUCCESS;
}
