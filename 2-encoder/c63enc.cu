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

extern "C" {
#include "c63.h"
#include "c63_write.h"
#include "common.h"
#include "common_sisci.h"
#include "me.h"
#include "tables.h"
}

static char *output_file;
FILE *outfile;

/* getopt */
extern int optind;
extern char *optarg;

static sci_error_t error;
static sci_desc_t sdY;
static sci_desc_t sdU;
static sci_desc_t sdV;
static sci_local_segment_t localSegmentY;
static sci_local_segment_t localSegmentU;
static sci_local_segment_t localSegmentV;
static sci_map_t localMapY;
static sci_map_t localMapU;
static sci_map_t localMapV;
static sci_local_data_interrupt_t interruptFromReader;
static sci_remote_interrupt_t interruptToReader;
static unsigned int interruptFromReaderNo;
static unsigned int localAdapterNo;
static unsigned int localNodeId;
static unsigned int localSegmentIdY;
static unsigned int localSegmentIdU;
static unsigned int localSegmentIdV;
static unsigned int readerNodeId;

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

	write_frame(cm);

	++cm->framenum;
	++cm->frames_since_keyframe;
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
	printf("  -o                             Output file (.c63)\n");
	printf("  -r                             Reader node ID\n");
	printf("\n");

	exit(EXIT_FAILURE);
}

#define SISCI_ERROR_CHECK

#ifdef SISCI_ERROR_CHECK
#define sisci_assert(error) { sisci_check_error(error, __FILE__, __LINE__, true); }
#define sisci_check(error) { sisci_check_error(error, __FILE__, __LINE__, false); }
#else
#define sisci_assert(error) {}
#define sisci_check(error) {}
#endif

inline static void sisci_check_error(sci_error_t error, const char* file, int line, bool terminate)
{
	if (error != SCI_ERR_OK)
	{
		fprintf(stderr, "SISCI error code 0x%x at %s, line %d", error, file, line);

		if (terminate)
		{
			SCITerminate();
			exit(EXIT_FAILURE);
		}
	}
}

static void init_SISCI()
{
	sci_error_t error;

	SCIInitialize(NO_FLAGS, &error);
	sisci_assert(error);

	SCIOpen(&sdY, NO_FLAGS, &error);
	sisci_assert(error);

	SCIOpen(&sdU, NO_FLAGS, &error);
	sisci_assert(error);

	SCIOpen(&sdV, NO_FLAGS, &error);
	sisci_assert(error);

	SCIGetLocalNodeId(localAdapterNo, &localNodeId, NO_FLAGS, &error);
	sisci_assert(error);

	// Interrupts from the reader
	interruptFromReaderNo = MORE_DATA_TRANSFERED;
	SCICreateDataInterrupt(sdY, &interruptFromReader, localAdapterNo, &interruptFromReaderNo, NULL,
			NULL, SCI_FLAG_FIXED_INTNO, &error);
	sisci_assert(error);

	// Interrupts to the reader
	printf("Connecting to interrupt on reader...\n");
	do
	{
		SCIConnectInterrupt(sdY, &interruptToReader, readerNodeId, localAdapterNo,
				READY_FOR_ORIG_TRANSFER, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
	}
	while (error != SCI_ERR_OK);
	printf("Done\n");
}

static void receive_width_and_height(int& width, int& height)
{
	printf("Waiting for widths and heights from reader...\n");
	uint32_t widthsAndHeights[8];
	unsigned int length = 8 * sizeof(uint32_t);
	SCIWaitForDataInterrupt(interruptFromReader, &widthsAndHeights, &length, SCI_INFINITE_TIMEOUT,
			NO_FLAGS, &error);
	sisci_assert(error);

	width = widthsAndHeights[0];
	height = widthsAndHeights[1];
}

static void init_SISCI_segments(struct c63_common* cm)
{
	localSegmentIdY = (localNodeId << 16) | readerNodeId | Y_COMPONENT;
	localSegmentIdU = (localNodeId << 16) | readerNodeId | U_COMPONENT;
	localSegmentIdV = (localNodeId << 16) | readerNodeId | V_COMPONENT;
	unsigned int segmentSizeY = cm->ypw * cm->yph * sizeof(uint8_t);
	unsigned int segmentSizeU = cm->upw * cm->uph * sizeof(uint8_t);
	unsigned int segmentSizeV = cm->vpw * cm->vph * sizeof(uint8_t);

	SCICreateSegment(sdY, &localSegmentY, localSegmentIdY, segmentSizeY, NO_CALLBACK, NULL,
			NO_FLAGS, &error);
	sisci_assert(error);

	SCICreateSegment(sdU, &localSegmentU, localSegmentIdU, segmentSizeU, NO_CALLBACK, NULL,
			NO_FLAGS, &error);
	sisci_assert(error);

	SCICreateSegment(sdV, &localSegmentV, localSegmentIdV, segmentSizeV, NO_CALLBACK, NULL,
			NO_FLAGS, &error);
	sisci_assert(error);

	SCIPrepareSegment(localSegmentY, localAdapterNo, NO_FLAGS, &error);
	sisci_assert(error);

	SCIPrepareSegment(localSegmentU, localAdapterNo, NO_FLAGS, &error);
	sisci_assert(error);

	SCIPrepareSegment(localSegmentV, localAdapterNo, NO_FLAGS, &error);
	sisci_assert(error);

	unsigned int offset = 0; // TODO: OK?

	image.Y = (uint8_t*) SCIMapLocalSegment(localSegmentY, &localMapY, offset, segmentSizeY, NULL, NO_FLAGS, &error);
	sisci_assert(error);

	image.U = (uint8_t*) SCIMapLocalSegment(localSegmentU, &localMapU, offset, segmentSizeU, NULL, NO_FLAGS, &error);
	sisci_assert(error);

	image.V = (uint8_t*) SCIMapLocalSegment(localSegmentV, &localMapV, offset, segmentSizeV, NULL, NO_FLAGS, &error);
	sisci_assert(error);

	SCISetSegmentAvailable(localSegmentY, localAdapterNo, NO_FLAGS, &error);
	sisci_assert(error);

	SCISetSegmentAvailable(localSegmentU, localAdapterNo, NO_FLAGS, &error);
	sisci_assert(error);

	SCISetSegmentAvailable(localSegmentV, localAdapterNo, NO_FLAGS, &error);
	sisci_assert(error);
}

static void cleanup_SISCI()
{
	sci_error_t error;

	SCIDisconnectInterrupt(interruptToReader, NO_FLAGS, &error);
	sisci_check(error);

	SCIRemoveDataInterrupt(interruptFromReader, NO_FLAGS, &error);
	sisci_check(error);

	SCISetSegmentUnavailable(localSegmentY, localAdapterNo, NO_FLAGS, &error);
	sisci_check(error);

	SCISetSegmentUnavailable(localSegmentU, localAdapterNo, NO_FLAGS, &error);
	sisci_check(error);

	SCISetSegmentUnavailable(localSegmentV, localAdapterNo, NO_FLAGS, &error);
	sisci_check(error);

	SCIUnmapSegment(localMapY, NO_FLAGS, &error);
	sisci_check(error);

	SCIUnmapSegment(localMapU, NO_FLAGS, &error);
	sisci_check(error);

	SCIUnmapSegment(localMapV, NO_FLAGS, &error);
	sisci_check(error);

	SCIRemoveSegment(localSegmentY, NO_FLAGS, &error);
	sisci_check(error);

	SCIRemoveSegment(localSegmentU, NO_FLAGS, &error);
	sisci_check(error);

	SCIRemoveSegment(localSegmentV, NO_FLAGS, &error);
	sisci_check(error);

	SCIClose(sdY, NO_FLAGS, &error);
	sisci_check(error);

	SCIClose(sdU, NO_FLAGS, &error);
	sisci_check(error);

	SCIClose(sdV, NO_FLAGS, &error);
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

	while ((c = getopt(argc, argv, "a:o:r:")) != -1)
	{
		switch (c)
		{
			case 'a':
				localAdapterNo = atoi(optarg);
				break;
			case 'o':
				output_file = optarg;
				break;
			case 'r':
				readerNodeId = atoi(optarg);
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

	outfile = fopen(output_file, "wb");

	if (outfile == NULL)
	{
		perror("fopen");
		exit(EXIT_FAILURE);
	}

	/* Encode input frames */
	int numframes = 0;

	init_SISCI();

	int width, height;
	receive_width_and_height(width, height);

	struct c63_common *cm = init_c63_enc(width, height);
	cm->e_ctx.fp = outfile;

	init_SISCI_segments(cm);

	uint8_t done;
	unsigned int done_size = sizeof(uint8_t);

	while (1)
	{
		printf("Waiting for interrupt...\n");
		SCIWaitForDataInterrupt(interruptFromReader, &done, &done_size, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
		sisci_assert(error);

		if (done)
		{
			printf("DONE from reader\n");
			break;
		}

		printf("Encoding frame %d, ", numframes);
		c63_encode_image(cm, &image);

		printf("Done!\n");

		++numframes;

		// Reader can transfer next frame
		SCITriggerInterrupt(interruptToReader, NO_FLAGS, &error);
		sisci_assert(error);
	}

	free_c63_enc(cm);
	fclose(outfile);

	cleanup_SISCI();

	return EXIT_SUCCESS;
}
