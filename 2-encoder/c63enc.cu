#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sisci_api.h"
#include "sisci_error.h"

#include "../common/sisci_common.h"
#include "../common/sisci_errchk.h"

extern "C" {
#include "c63.h"
#include "common.h"
#include "me.h"
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

static void c63_encode_image(struct c63_common *cm, yuv_t *image)
{
	/* Advance to next frame */
	destroy_frame(cm->refframe);
	cm->refframe = cm->curframe;
	cm->curframe = create_frame(cm, image);

	/* Check if keyframe */
	if (cm->framenum == 0 || cm->frames_since_keyframe == cm->keyframe_interval)
	{
		cm->curframe->keyframe = 1;
		cm->frames_since_keyframe = 0;

		fprintf(stderr, " (keyframe) ");
	}
	else
	{
		cm->curframe->keyframe = 0;
	}

	if (!cm->curframe->keyframe)
	{
		/* Motion Estimation */
		c63_motion_estimate(cm);

		/* Motion Compensation */
		c63_motion_compensate(cm);
	}

	/* DCT and Quantization */
	dct_quantize(image->Y, cm->curframe->predicted->Y, cm->padw[Y_COMPONENT], cm->padh[Y_COMPONENT],
			cm->curframe->residuals->Ydct, cm->quanttbl[Y_COMPONENT]);

	dct_quantize(image->U, cm->curframe->predicted->U, cm->padw[U_COMPONENT], cm->padh[U_COMPONENT],
			cm->curframe->residuals->Udct, cm->quanttbl[U_COMPONENT]);

	dct_quantize(image->V, cm->curframe->predicted->V, cm->padw[V_COMPONENT], cm->padh[V_COMPONENT],
			cm->curframe->residuals->Vdct, cm->quanttbl[V_COMPONENT]);

	/* Reconstruct frame for inter-prediction */
	dequantize_idct(cm->curframe->residuals->Ydct, cm->curframe->predicted->Y, cm->ypw, cm->yph,
			cm->curframe->recons->Y, cm->quanttbl[Y_COMPONENT]);
	dequantize_idct(cm->curframe->residuals->Udct, cm->curframe->predicted->U, cm->upw, cm->uph,
			cm->curframe->recons->U, cm->quanttbl[U_COMPONENT]);
	dequantize_idct(cm->curframe->residuals->Vdct, cm->curframe->predicted->V, cm->vpw, cm->vph,
			cm->curframe->recons->V, cm->quanttbl[V_COMPONENT]);

	/* Function dump_image(), found in common.c, can be used here to check if the
	 prediction is correct */
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

	cm->mb_cols = cm->ypw / 8;
	cm->mb_rows = cm->yph / 8;

	/* Quality parameters -- Home exam deliveries should have original values,
	 i.e., quantization factor should be 25, search range should be 16, and the
	 keyframe interval should be 100. */
	cm->qp = 25;                  // Constant quantization factor. Range: [1..50]
	cm->me_search_range = 16;     // Pixels in every direction
	cm->keyframe_interval = 100;  // Distance between keyframes

	/* Initialize quantization tables */
	for (int i = 0; i < 64; ++i)
	{
		cm->quanttbl[Y_COMPONENT][i] = yquanttbl_def[i] / (cm->qp / 10.0);
		cm->quanttbl[U_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
		cm->quanttbl[V_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
	}

	return cm;
}

void free_c63_enc(struct c63_common* cm)
{
	destroy_frame(cm->curframe);
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
	printf("Waiting for widths and heights from reader... ");
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
	residuals_offset_Y = mb_offset_Y + cm->mb_rows*cm->mb_cols*sizeof(struct macroblock);
	mb_offset_U = residuals_offset_Y + cm->ypw*cm->yph*sizeof(int16_t);
	residuals_offset_U = mb_offset_U + (cm->mb_rows/2)*(cm->mb_cols/2)*sizeof(struct macroblock);
	mb_offset_V = residuals_offset_U + cm->upw*cm->uph*sizeof(int16_t);
	residuals_offset_V = mb_offset_V +  (cm->mb_rows/2)*(cm->mb_cols/2)*sizeof(struct macroblock);
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
	printf("her");

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

	while (1)
	{
		printf("Waiting for interrupt...\n");
		SCIWaitForDataInterrupt(interruptFromReader, &done, &done_size, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
		sisci_assert(error);

		if (done)
		{
			printf("DONE from reader\n");
			// Send interrupt to writer signalling that encoding has been finished
			SCITriggerDataInterrupt(interruptToWriter, (void*) &done, sizeof(uint8_t), SCI_NO_FLAGS, &error);
			sisci_assert(error);
			break;
		}

		printf("Encoding frame %d, ", numframes);
		c63_encode_image(cm, &image);

		if (numframes != 0) {
			// Wait for interrupt from writer
			do {
				SCIWaitForInterrupt(interruptFromWriter, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
			} while (error != SCI_ERR_OK);
		}

		// Copy data frame to remote segment
		printf("Sending frame %d to writer\n", numframes);

		SCIMemCpy(writer_sequence, (void*) &cm->curframe->keyframe, remoteMap, keyframe_offset, sizeof(int), SCI_FLAG_ERROR_CHECK, &error);
		sisci_assert(error);

		SCIMemCpy(writer_sequence, cm->curframe->mbs[Y_COMPONENT], remoteMap, mb_offset_Y, cm->mb_rows*cm->mb_cols*sizeof(struct macroblock), SCI_FLAG_ERROR_CHECK, &error);
		sisci_assert(error);
		SCIMemCpy(writer_sequence, cm->curframe->residuals->Ydct, remoteMap, residuals_offset_Y, cm->ypw*cm->yph*sizeof(int16_t), SCI_FLAG_ERROR_CHECK, &error);
		sisci_assert(error);

		SCIMemCpy(writer_sequence, cm->curframe->mbs[U_COMPONENT], remoteMap, mb_offset_U, (cm->mb_rows/2)*(cm->mb_cols/2)*sizeof(struct macroblock), SCI_FLAG_ERROR_CHECK, &error);
		sisci_assert(error);
		SCIMemCpy(writer_sequence, cm->curframe->residuals->Udct, remoteMap, residuals_offset_U, cm->upw*cm->uph*sizeof(int16_t), SCI_FLAG_ERROR_CHECK, &error);
		sisci_assert(error);

		SCIMemCpy(writer_sequence, cm->curframe->mbs[V_COMPONENT], remoteMap, mb_offset_V, (cm->mb_rows/2)*(cm->mb_cols/2)*sizeof(struct macroblock), SCI_FLAG_ERROR_CHECK, &error);
		sisci_assert(error);
		SCIMemCpy(writer_sequence, cm->curframe->residuals->Vdct, remoteMap, residuals_offset_V, cm->vpw*cm->vph*sizeof(int16_t), SCI_FLAG_ERROR_CHECK, &error);
		sisci_assert(error);

		printf("Done!\n");

		// Send interrupt to writer signalling the data has been transfered
		SCITriggerDataInterrupt(interruptToWriter, (void*) &done, sizeof(uint8_t), SCI_NO_FLAGS, &error);
		sisci_assert(error);

		++cm->framenum;
		++cm->frames_since_keyframe;

		printf("Done!\n");

		++numframes;

		// Reader can transfer next frame
		SCITriggerInterrupt(interruptToReader, SCI_NO_FLAGS, &error);
		sisci_assert(error);
	}

	free_c63_enc(cm);

	cleanup_SISCI();

	return EXIT_SUCCESS;
}
