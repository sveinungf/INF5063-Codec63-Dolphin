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

#include "c63.h"
#include "common.h"
#include "me.h"
#include "sisci.h"

extern "C" {
#include "tables.h"
}


/* getopt */
extern int optind;
extern char *optarg;

static void zero_out_prediction(struct c63_common* cm)
{
	struct frame* frame = cm->curframe;
	cudaMemsetAsync(frame->predicted_gpu->Y, 0, cm->ypw * cm->yph * sizeof(uint8_t), cm->cuda_data.streamY);
	cudaMemsetAsync(frame->predicted_gpu->U, 0, cm->upw * cm->uph * sizeof(uint8_t), cm->cuda_data.streamU);
	cudaMemsetAsync(frame->predicted_gpu->V, 0, cm->vpw * cm->vph * sizeof(uint8_t), cm->cuda_data.streamV);
}

static void c63_encode_image(struct c63_common *cm, struct segment_yuv* image_gpu)
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
	else { cm->curframe->keyframe = 0; }

	if (!cm->curframe->keyframe)
	{
		/* Motion Estimation */
		gpu_c63_motion_estimate(cm);

		/* Motion Compensation */
		gpu_c63_motion_compensate(cm);
	}
	else
	{
		// dct_quantize() expects zeroed out prediction buffers for key frames.
		// We zero them out here since we reuse the buffers from previous frames.
		zero_out_prediction(cm);
	}

	yuv_t* predicted = cm->curframe->predicted_gpu;
	dct_t* residuals = cm->curframe->residuals_gpu;

	const dim3 threadsPerBlock(8, 8);

	const dim3 numBlocks_Y(cm->padw[Y_COMPONENT]/threadsPerBlock.x, cm->padh[Y_COMPONENT]/threadsPerBlock.y);
	const dim3 numBlocks_UV(cm->padw[U_COMPONENT]/threadsPerBlock.x, cm->padh[U_COMPONENT]/threadsPerBlock.y);

	/* DCT and Quantization */
	dct_quantize<<<numBlocks_Y, threadsPerBlock, 0, cm->cuda_data.streamY>>>((const uint8_t*)cm->curframe->orig_gpu->Y, predicted->Y,
			cm->padw[Y_COMPONENT], residuals->Ydct, Y_COMPONENT);
	cudaMemcpyAsync(cm->curframe->residuals->Ydct, residuals->Ydct, cm->padw[Y_COMPONENT]*cm->padh[Y_COMPONENT]*sizeof(int16_t),
			cudaMemcpyDeviceToHost, cm->cuda_data.streamY);

	dct_quantize<<<numBlocks_UV, threadsPerBlock, 0, cm->cuda_data.streamU>>>((const uint8_t*)cm->curframe->orig_gpu->U, predicted->U,
			cm->padw[U_COMPONENT], residuals->Udct, U_COMPONENT);
	cudaMemcpyAsync(cm->curframe->residuals->Udct, residuals->Udct, cm->padw[U_COMPONENT]*cm->padh[U_COMPONENT]*sizeof(int16_t),
			cudaMemcpyDeviceToHost, cm->cuda_data.streamU);

	dct_quantize<<<numBlocks_UV, threadsPerBlock, 0, cm->cuda_data.streamV>>>((const uint8_t*)cm->curframe->orig_gpu->V, predicted->V,
			cm->padw[V_COMPONENT], residuals->Vdct, V_COMPONENT);
	cudaMemcpyAsync(cm->curframe->residuals->Vdct, residuals->Vdct, cm->padw[V_COMPONENT]*cm->padh[V_COMPONENT]*sizeof(int16_t),
			cudaMemcpyDeviceToHost, cm->cuda_data.streamV);

	/* Reconstruct frame for inter-prediction */
	dequantize_idct<<<numBlocks_Y, threadsPerBlock, 0, cm->cuda_data.streamY>>>(residuals->Ydct, predicted->Y,
			cm->ypw, cm->curframe->recons_gpu->Y, Y_COMPONENT);

	dequantize_idct<<<numBlocks_UV, threadsPerBlock, 0, cm->cuda_data.streamU>>>(residuals->Udct, predicted->U,
			cm->upw, cm->curframe->recons_gpu->U, U_COMPONENT);

	dequantize_idct<<<numBlocks_UV, threadsPerBlock, 0, cm->cuda_data.streamV>>>(residuals->Vdct, predicted->V,
			cm->vpw, cm->curframe->recons_gpu->V, V_COMPONENT);

	/* Function dump_image(), found in common.c, can be used here to check if the
     prediction is correct */
}

static void init_boundaries(c63_common* cm)
{
	int hY = cm->padh[Y_COMPONENT];
	int hUV = cm->padh[U_COMPONENT];

	int wY = cm->padw[Y_COMPONENT];
	int wUV = cm->padw[U_COMPONENT];

	int* leftsY = new int[cm->mb_colsY];
	int* leftsUV = new int[cm->mb_colsUV];
	int* rightsY = new int[cm->mb_colsY];
	int* rightsUV = new int[cm->mb_colsUV];
	int* topsY = new int[cm->mb_rowsY];
	int* topsUV = new int[cm->mb_rowsUV];
	int* bottomsY = new int[cm->mb_rowsY];
	int* bottomsUV = new int[cm->mb_rowsUV];

	for (int mb_x = 0; mb_x < cm->mb_colsY; ++mb_x) {
		leftsY[mb_x] = mb_x * 8 - ME_RANGE_Y;
		rightsY[mb_x] = mb_x * 8 + ME_RANGE_Y;

		if (leftsY[mb_x] < 0) {
			leftsY[mb_x] = 0;
		}

		if (rightsY[mb_x] > (wY - 8)) {
			rightsY[mb_x] = wY - 8;
		}
	}

	for (int mb_x = 0; mb_x < cm->mb_colsUV; ++mb_x) {
		leftsUV[mb_x] = mb_x * 8 - ME_RANGE_UV;
		rightsUV[mb_x] = mb_x * 8 + ME_RANGE_UV;

		if (leftsUV[mb_x] < 0) {
			leftsUV[mb_x] = 0;
		}

		if (rightsUV[mb_x] > (wUV - 8)) {
			rightsUV[mb_x] = wUV - 8;
		}
	}

	for (int mb_y = 0; mb_y < cm->mb_rowsY; ++mb_y) {
		topsY[mb_y] = mb_y * 8 - ME_RANGE_Y;
		bottomsY[mb_y] = mb_y * 8 + ME_RANGE_Y;

		if (topsY[mb_y] < 0) {
			topsY[mb_y] = 0;
		}

		if (bottomsY[mb_y] > (hY - 8)) {
			bottomsY[mb_y] = hY - 8;
		}
	}

	for (int mb_y = 0; mb_y < cm->mb_rowsUV; ++mb_y) {
		topsUV[mb_y] = mb_y * 8 - ME_RANGE_UV;
		bottomsUV[mb_y] = mb_y * 8 + ME_RANGE_UV;

		if (topsUV[mb_y] < 0) {
			topsUV[mb_y] = 0;
		}

		if (bottomsUV[mb_y] > (hUV - 8)) {
			bottomsUV[mb_y] = hUV - 8;
		}
	}

	struct boundaries* boundY = &cm->me_boundariesY;
	cudaMalloc((void**) &boundY->left, cm->mb_colsY * sizeof(int));
	cudaMalloc((void**) &boundY->right, cm->mb_colsY * sizeof(int));
	cudaMalloc((void**) &boundY->top, cm->mb_rowsY * sizeof(int));
	cudaMalloc((void**) &boundY->bottom, cm->mb_rowsY * sizeof(int));

	struct boundaries* boundUV = &cm->me_boundariesUV;
	cudaMalloc((void**) &boundUV->left, cm->mb_colsUV * sizeof(int));
	cudaMalloc((void**) &boundUV->right, cm->mb_colsUV * sizeof(int));
	cudaMalloc((void**) &boundUV->top, cm->mb_rowsUV * sizeof(int));
	cudaMalloc((void**) &boundUV->bottom, cm->mb_rowsUV * sizeof(int));

	const cudaStream_t& streamY = cm->cuda_data.streamY;
	cudaMemcpyAsync((void*) boundY->left, leftsY, cm->mb_colsY * sizeof(int), cudaMemcpyHostToDevice, streamY);
	cudaMemcpyAsync((void*) boundY->right, rightsY, cm->mb_colsY * sizeof(int), cudaMemcpyHostToDevice, streamY);
	cudaMemcpyAsync((void*) boundY->top, topsY, cm->mb_rowsY * sizeof(int), cudaMemcpyHostToDevice, streamY);
	cudaMemcpyAsync((void*) boundY->bottom, bottomsY, cm->mb_rowsY * sizeof(int), cudaMemcpyHostToDevice, streamY);

	cudaMemcpy((void*) boundUV->left, leftsUV, cm->mb_colsUV * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy((void*) boundUV->right, rightsUV, cm->mb_colsUV * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy((void*) boundUV->top, topsUV, cm->mb_rowsUV * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy((void*) boundUV->bottom, bottomsUV, cm->mb_rowsUV * sizeof(int), cudaMemcpyHostToDevice);

	delete[] leftsY;
	delete[] leftsUV;
	delete[] rightsY;
	delete[] rightsUV;
	delete[] topsY;
	delete[] topsUV;
	delete[] bottomsY;
	delete[] bottomsUV;
}

static void deinit_boundaries(c63_common* cm)
{
	cudaFree((void*) cm->me_boundariesY.left);
	cudaFree((void*) cm->me_boundariesY.right);
	cudaFree((void*) cm->me_boundariesY.top);
	cudaFree((void*) cm->me_boundariesY.bottom);

	cudaFree((void*) cm->me_boundariesUV.left);
	cudaFree((void*) cm->me_boundariesUV.right);
	cudaFree((void*) cm->me_boundariesUV.top);
	cudaFree((void*) cm->me_boundariesUV.bottom);
}

static void init_cuda_data(c63_common* cm)
{
	cuda_data* cuda_me = &(cm->cuda_data);

	cudaStreamCreate(&cuda_me->streamY);
	cudaStreamCreate(&cuda_me->streamU);
	cudaStreamCreate(&cuda_me->streamV);

	cudaMalloc((void**) &cuda_me->sad_index_resultsY, cm->mb_colsY*cm->mb_rowsY*sizeof(unsigned int));
	cudaMalloc((void**) &cuda_me->sad_index_resultsU, cm->mb_colsUV*cm->mb_rowsUV*sizeof(unsigned int));
	cudaMalloc((void**) &cuda_me->sad_index_resultsV, cm->mb_colsUV*cm->mb_rowsUV*sizeof(unsigned int));
}

static void deinit_cuda_data(c63_common* cm)
{
	cudaStreamDestroy(cm->cuda_data.streamY);
	cudaStreamDestroy(cm->cuda_data.streamU);
	cudaStreamDestroy(cm->cuda_data.streamV);

	cudaFree(cm->cuda_data.sad_index_resultsY);
	cudaFree(cm->cuda_data.sad_index_resultsU);
	cudaFree(cm->cuda_data.sad_index_resultsV);
}

/*
static void copy_image_to_gpu(struct c63_common* cm, const struct segment_yuv& image, yuv_t* image_gpu)
{
	cudaMemcpyAsync(image_gpu->Y, (void*) image.Y, cm->ypw * cm->yph * sizeof(uint8_t), cudaMemcpyHostToDevice, cm->cuda_data.streamY);
	cudaMemcpyAsync(image_gpu->U, (void*) image.U, cm->upw * cm->uph * sizeof(uint8_t), cudaMemcpyHostToDevice, cm->cuda_data.streamU);
	cudaMemcpyAsync(image_gpu->V, (void*) image.V, cm->vpw * cm->vph * sizeof(uint8_t), cudaMemcpyHostToDevice, cm->cuda_data.streamV);
}
*/

struct c63_common* init_c63_enc(int width, int height)
{
	/* calloc() sets allocated memory to zero */
	struct c63_common *cm = (struct c63_common*) calloc(1, sizeof(struct c63_common));

	cm->width = width;
	cm->height = height;

	cm->padw[Y_COMPONENT] = cm->ypw = (uint32_t) (ceil(width / 16.0f) * 16);
	cm->padh[Y_COMPONENT] = cm->yph = (uint32_t) (ceil(height / 16.0f) * 16);
	cm->padw[U_COMPONENT] = cm->upw = (uint32_t) (ceil(width * UX / (YX * 8.0f)) * 8);
	cm->padh[U_COMPONENT] = cm->uph = (uint32_t) (ceil(height * UY / (YY * 8.0f)) * 8);
	cm->padw[V_COMPONENT] = cm->vpw = (uint32_t) (ceil(width * VX / (YX * 8.0f)) * 8);
	cm->padh[V_COMPONENT] = cm->vph = (uint32_t) (ceil(height * VY / (YY * 8.0f)) * 8);

	cm->mb_colsY = cm->ypw / 8;
	cm->mb_rowsY = cm->yph / 8;
	cm->mb_colsUV = cm->mb_colsY / 2;
	cm->mb_rowsUV = cm->mb_rowsY / 2;

	/* Quality parameters -- Home exam deliveries should have original values,
	 i.e., quantization factor should be 25, search range should be 16, and the
	 keyframe interval should be 100. */
	cm->qp = 25;                  // Constant quantization factor. Range: [1..50]
	//cm->me_search_range = 16;   // This is now defined in c63.h
	cm->keyframe_interval = 100;  // Distance between keyframes

	/* Initialize quantization tables */
	for (int i = 0; i < 64; ++i)
	{
		cm->quanttbl[Y_COMPONENT][i] = yquanttbl_def[i] / (cm->qp / 10.0);
		cm->quanttbl[U_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
		cm->quanttbl[V_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
	}

	init_cuda_data(cm);

	cm->curframe = create_frame(cm);
	cm->refframe = create_frame(cm);

	init_boundaries(cm);

	return cm;
}

void free_c63_enc(struct c63_common* cm)
{
	deinit_boundaries(cm);

	destroy_frame(cm->curframe);
	destroy_frame(cm->refframe);

	deinit_cuda_data(cm);

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

	struct c63_common *cm = init_c63_enc(width, height);

	set_sizes_offsets(cm);

	struct segment_yuv images_gpu[2];
	images_gpu[0] = init_image_segment(cm, 0);
	images_gpu[1] = init_image_segment(cm, 1);
	init_remote_encoded_data_segment(0);
	init_remote_encoded_data_segment(1);
	init_local_encoded_data_segments();

	//yuv_t* image_gpu = create_image_gpu(cm);
	int segNum = 0;

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

		c63_encode_image(cm, &images_gpu[segNum]);

		// Wait until the GPU has finished encoding
		cudaStreamSynchronize(cm->cuda_data.streamY);
		cudaStreamSynchronize(cm->cuda_data.streamU);
		cudaStreamSynchronize(cm->cuda_data.streamV);

		printf(", encoded\n");
		fflush(stdout);

		if (numframes != 0 && transferred == NUM_IMAGE_SEGMENTS) {
			// The writer sends an interrupt when it is ready for the next frame
			wait_for_writer(segNum);
			--transferred;
		}

		// Copy data frame to remote segment - interrupt to writer handled by callback
		transfer_encoded_data(cm->curframe->keyframe, cm->curframe->mbs, cm->curframe->residuals, segNum);
		++transferred;

		// Reader can transfer next frame
		signal_reader(segNum);

		// Send interrupt to writer signaling the data has been transfered
		//signal_writer(DATA_TRANSFERRED);

		++cm->framenum;
		++cm->frames_since_keyframe;

		++numframes;

		segNum ^= 1;
	}

	//destroy_image_gpu(image_gpu);

	free_c63_enc(cm);

	cleanup_segments();
	cleanup_SISCI();

	cudaDeviceReset();

	return EXIT_SUCCESS;
}
