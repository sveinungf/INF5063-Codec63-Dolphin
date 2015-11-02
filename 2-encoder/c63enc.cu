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
#include "sisci_api.h"
#include "sisci_error.h"

#include "../common/sisci_common.h"
#include "../common/sisci_errchk.h"
#include "c63.h"
#include "common.h"
#include "me.h"

extern "C" {
#include "tables.h"
}


/* getopt */
extern int optind;
extern char *optarg;

static sci_error_t error;
static sci_desc_t sd;

static sci_local_segment_t localSegment;
static sci_map_t localMap;

static sci_remote_segment_t remoteSegment;
static sci_map_t remoteMap;

static sci_local_data_interrupt_t interruptFromReader;
static sci_remote_interrupt_t interruptToReader;

static sci_local_interrupt_t interruptFromWriter;
static sci_remote_data_interrupt_t interruptToWriter;

static unsigned int interruptFromReaderNo;
static unsigned int interruptFromWriterNo;

static unsigned int localAdapterNo;
static unsigned int localNodeId;
static unsigned int readerNodeId;
static unsigned int writerNodeId;

static sci_sequence_t writer_sequence;

static uint32_t mb_offset_Y;
static uint32_t mb_offset_U;
static uint32_t mb_offset_V;

static uint32_t keyframe_offset;

static uint32_t residuals_offset_Y;
static uint32_t residuals_offset_U;
static uint32_t residuals_offset_V;

static yuv_t image;

static void zero_out_prediction(struct c63_common* cm)
{
	struct frame* frame = cm->curframe;
	cudaMemsetAsync(frame->predicted_gpu->Y, 0, cm->ypw * cm->yph * sizeof(uint8_t), cm->cuda_data.streamY);
	cudaMemsetAsync(frame->predicted_gpu->U, 0, cm->upw * cm->uph * sizeof(uint8_t), cm->cuda_data.streamU);
	cudaMemsetAsync(frame->predicted_gpu->V, 0, cm->vpw * cm->vph * sizeof(uint8_t), cm->cuda_data.streamV);
}

static void c63_encode_image(struct c63_common *cm, yuv_t* image_gpu)
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
		c63_motion_estimate(cm);

		/* Motion Compensation */
		c63_motion_compensate(cm);
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
	dct_quantize<<<numBlocks_Y, threadsPerBlock, 0, cm->cuda_data.streamY>>>(cm->curframe->orig_gpu->Y, predicted->Y,
			cm->padw[Y_COMPONENT], residuals->Ydct, Y_COMPONENT);
	cudaMemcpyAsync(cm->curframe->residuals->Ydct, residuals->Ydct, cm->padw[Y_COMPONENT]*cm->padh[Y_COMPONENT]*sizeof(int16_t),
			cudaMemcpyDeviceToHost, cm->cuda_data.streamY);

	dct_quantize<<<numBlocks_UV, threadsPerBlock, 0, cm->cuda_data.streamU>>>(cm->curframe->orig_gpu->U, predicted->U,
			cm->padw[U_COMPONENT], residuals->Udct, U_COMPONENT);
	cudaMemcpyAsync(cm->curframe->residuals->Udct, residuals->Udct, cm->padw[U_COMPONENT]*cm->padh[U_COMPONENT]*sizeof(int16_t),
			cudaMemcpyDeviceToHost, cm->cuda_data.streamU);

	dct_quantize<<<numBlocks_UV, threadsPerBlock, 0, cm->cuda_data.streamV>>>(cm->curframe->orig_gpu->V, predicted->V,
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

static void copy_image_to_gpu(struct c63_common* cm, yuv_t* image, yuv_t* image_gpu)
{
	cudaMemcpyAsync(image_gpu->Y, image->Y, cm->ypw * cm->yph * sizeof(uint8_t), cudaMemcpyHostToDevice, cm->cuda_data.streamY);
	cudaMemcpyAsync(image_gpu->U, image->U, cm->upw * cm->uph * sizeof(uint8_t), cudaMemcpyHostToDevice, cm->cuda_data.streamU);
	cudaMemcpyAsync(image_gpu->V, image->V, cm->vpw * cm->vph * sizeof(uint8_t), cudaMemcpyHostToDevice, cm->cuda_data.streamV);
}

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

static void init_SISCI()
{
	sci_error_t error;

	SCIInitialize(SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCIOpen(&sd, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCIGetLocalNodeId(localAdapterNo, &localNodeId, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	// Interrupts from the reader
	interruptFromReaderNo = MORE_DATA_TRANSFERRED;
	SCICreateDataInterrupt(sd, &interruptFromReader, localAdapterNo, &interruptFromReaderNo, NULL,
			NULL, SCI_FLAG_FIXED_INTNO, &error);
	sisci_assert(error);

	// Interrupts to the reader
	printf("Connecting to interrupt on reader... ");
	fflush(stdout);
	do
	{
		SCIConnectInterrupt(sd, &interruptToReader, readerNodeId, localAdapterNo,
				READY_FOR_ORIG_TRANSFER, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	}
	while (error != SCI_ERR_OK);
	printf("Done!\n");


	// Interrupts from the writer
	interruptFromWriterNo = DATA_WRITTEN;
	SCICreateInterrupt(sd, &interruptFromWriter, localAdapterNo, &interruptFromWriterNo, NULL,
			NULL, SCI_FLAG_FIXED_INTNO, &error);
	sisci_assert(error);

	// Interrupts to the writer
	printf("Connecting to interrupt on writer... ");
	fflush(stdout);
	do
	{
		SCIConnectDataInterrupt(sd, &interruptToWriter, writerNodeId, localAdapterNo,
				ENCODED_FRAME_TRANSFERRED, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	}
	while (error != SCI_ERR_OK);
	printf("Done!\n");
}

static void receive_width_and_height(int& width, int& height)
{
	printf("Waiting for width and height from reader... ");
	fflush(stdout);

	uint32_t widthAndHeight[2];
	unsigned int length = 2 * sizeof(uint32_t);
	SCIWaitForDataInterrupt(interruptFromReader, &widthAndHeight, &length, SCI_INFINITE_TIMEOUT,
			SCI_NO_FLAGS, &error);
	sisci_assert(error);

	width = widthAndHeight[0];
	height = widthAndHeight[1];
	printf("Done!\n");
}

static void send_width_and_height(uint32_t width, uint32_t height) {
	uint32_t widthAndHeight[2] = {width, height};
	SCITriggerDataInterrupt(interruptToWriter, (void*) &widthAndHeight, 2*sizeof(uint32_t), SCI_NO_FLAGS, &error);
	sisci_assert(error);
}

static void init_SISCI_segments(struct c63_common* cm)
{
	unsigned int localSegmentId = (localNodeId << 16) | (readerNodeId << 8) | 0;

	unsigned int segmentSizeY = cm->ypw * cm->yph * sizeof(uint8_t);
	unsigned int segmentSizeU = cm->upw * cm->uph * sizeof(uint8_t);
	unsigned int segmentSizeV = cm->vpw * cm->vph * sizeof(uint8_t);

	unsigned int segmentSize = segmentSizeY + segmentSizeU + segmentSizeV;

	SCICreateSegment(sd, &localSegment, localSegmentId, segmentSize, SCI_NO_CALLBACK, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCIPrepareSegment(localSegment, localAdapterNo, SCI_NO_FLAGS, &error);

	void* buffer = SCIMapLocalSegment(localSegment, &localMap, 0, segmentSize, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	unsigned int offset = 0;

	image.Y = (uint8_t*) buffer + offset;
	offset += segmentSizeY;
	image.U = (uint8_t*) buffer + offset;
	offset += segmentSizeU;
	image.V = (uint8_t*) buffer + offset;
	offset += segmentSizeV;

	SCISetSegmentAvailable(localSegment, localAdapterNo, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	// Connect to remote segment on writer
	unsigned int remoteSegmentId = (writerNodeId << 16) | (localNodeId) | 0;

	do {
		SCIConnectSegment(sd, &remoteSegment, writerNodeId, remoteSegmentId, localAdapterNo,
				SCI_NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	} while (error != SCI_ERR_OK);

	// Set segment size
	uint32_t remoteSegmentSize = SCIGetRemoteSegmentSize(remoteSegment);

	unsigned int offsett = 0;
	SCIMapRemoteSegment(remoteSegment, &remoteMap, offsett, remoteSegmentSize, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);

    SCICreateMapSequence(remoteMap, &writer_sequence, 0, &error);
    sisci_assert(error);

	// Set offsets within segment
	keyframe_offset = 0;
	mb_offset_Y = keyframe_offset + sizeof(int);
	residuals_offset_Y = mb_offset_Y + cm->mb_rowsY*cm->mb_colsY*sizeof(struct macroblock);
	mb_offset_U = residuals_offset_Y + cm->ypw*cm->yph*sizeof(int16_t);
	residuals_offset_U = mb_offset_U + cm->mb_rowsUV*cm->mb_colsUV*sizeof(struct macroblock);
	mb_offset_V = residuals_offset_U + cm->upw*cm->uph*sizeof(int16_t);
	residuals_offset_V = mb_offset_V + cm->mb_rowsUV*cm->mb_colsUV*sizeof(struct macroblock);
}

static void cleanup_SISCI()
{
	sci_error_t error;

	SCIDisconnectInterrupt(interruptToReader, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCIRemoveDataInterrupt(interruptFromReader, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCIDisconnectDataInterrupt(interruptToWriter, SCI_NO_FLAGS, &error);
	sisci_check(error);

	do {
		SCIRemoveInterrupt(interruptFromWriter, SCI_NO_FLAGS, &error);
	} while (error != SCI_ERR_OK);

	SCISetSegmentUnavailable(localSegment, localAdapterNo, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCIUnmapSegment(localMap, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCIRemoveSegment(localSegment, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCIUnmapSegment(remoteMap, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCIDisconnectSegment(remoteSegment, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCIClose(sd, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCITerminate();
}

int main(int argc, char **argv)
{
	int c;

	if (argc == 1)
	{
		print_help();
	}

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

	init_SISCI();

	int width, height;
	receive_width_and_height(width, height);
	send_width_and_height(width, height);

	struct c63_common *cm = init_c63_enc(width, height);

	init_SISCI_segments(cm);

	uint8_t done;
	unsigned int done_size = sizeof(uint8_t);

	yuv_t* image_gpu = create_image_gpu(cm);

	while (1)
	{
		printf("Frame %d:", numframes);
		fflush(stdout);

		// The reader sends an interrupt when it has transferred the next frame
		SCIWaitForDataInterrupt(interruptFromReader, &done, &done_size, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
		sisci_assert(error);

		if (!done)
		{
			printf(" Received");
			fflush(stdout);
		}
		else
		{
			printf("\rNo more frames from reader\n");

			// Send interrupt to writer signaling that encoding has been finished
			SCITriggerDataInterrupt(interruptToWriter, (void*) &done, sizeof(uint8_t), SCI_NO_FLAGS, &error);
			sisci_assert(error);
			break;
		}

		copy_image_to_gpu(cm, &image, image_gpu);

		c63_encode_image(cm, image_gpu);

		// Wait until the GPU has finished encoding
		cudaStreamSynchronize(cm->cuda_data.streamY);
		cudaStreamSynchronize(cm->cuda_data.streamU);
		cudaStreamSynchronize(cm->cuda_data.streamV);

		printf(", encoded");
		fflush(stdout);

		if (numframes != 0) {
			// The writer sends an interrupt when it is ready for the next frame
			do {
				SCIWaitForInterrupt(interruptFromWriter, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
			} while (error != SCI_ERR_OK);
		}

		// Copy data frame to remote segment
		// TODO: These currently fail when using SCI_FLAG_ERROR_CHECK
		SCIMemCpy(writer_sequence, (void*) &cm->curframe->keyframe, remoteMap, keyframe_offset, sizeof(int), SCI_NO_FLAGS, &error);
		sisci_assert(error);

		SCIMemCpy(writer_sequence, cm->curframe->mbs[Y_COMPONENT], remoteMap, mb_offset_Y, cm->mb_rowsY*cm->mb_colsY*sizeof(struct macroblock), SCI_NO_FLAGS, &error);
		sisci_assert(error);

		SCIMemCpy(writer_sequence, cm->curframe->residuals->Ydct, remoteMap, residuals_offset_Y, cm->ypw*cm->yph*sizeof(int16_t), SCI_NO_FLAGS, &error);
		sisci_assert(error);

		SCIMemCpy(writer_sequence, cm->curframe->mbs[U_COMPONENT], remoteMap, mb_offset_U, (cm->mb_rowsUV)*(cm->mb_colsUV)*sizeof(struct macroblock), SCI_NO_FLAGS, &error);
		sisci_assert(error);

		SCIMemCpy(writer_sequence, cm->curframe->residuals->Udct, remoteMap, residuals_offset_U, cm->upw*cm->uph*sizeof(int16_t), SCI_NO_FLAGS, &error);
		sisci_assert(error);

		SCIMemCpy(writer_sequence, cm->curframe->mbs[V_COMPONENT], remoteMap, mb_offset_V, (cm->mb_rowsUV)*(cm->mb_colsUV)*sizeof(struct macroblock), SCI_NO_FLAGS, &error);
		sisci_assert(error);

		SCIMemCpy(writer_sequence, cm->curframe->residuals->Vdct, remoteMap, residuals_offset_V, cm->vpw*cm->vph*sizeof(int16_t), SCI_NO_FLAGS, &error);
		sisci_assert(error);

		printf(", sent\n");

		// Send interrupt to writer signaling the data has been transfered
		SCITriggerDataInterrupt(interruptToWriter, (void*) &done, sizeof(uint8_t), SCI_NO_FLAGS, &error);
		sisci_assert(error);

		++cm->framenum;
		++cm->frames_since_keyframe;

		++numframes;

		// Reader can transfer next frame
		SCITriggerInterrupt(interruptToReader, SCI_NO_FLAGS, &error);
		sisci_assert(error);
	}

	destroy_image_gpu(image_gpu);

	free_c63_enc(cm);

	cleanup_SISCI();

	return EXIT_SUCCESS;
}
