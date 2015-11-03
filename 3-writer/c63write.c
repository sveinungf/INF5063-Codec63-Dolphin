#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/sisci_common.h"
#include "c63.h"
#include "c63_write.h"
#include "tables.h"

#include "sisci_api.h"

// SISCI variables
static sci_desc_t sd;

static unsigned int localAdapterNo;
static unsigned int localNodeId;
static unsigned int encoderNodeId;

static sci_local_segment_t localSegment;
static volatile uint8_t *local_buffer;
static sci_map_t localMap;

static sci_local_data_interrupt_t interruptFromEncoder;
static sci_remote_interrupt_t interruptToEncoder;

static unsigned int interruptFromEncoderNo;

static uint32_t keyframe_offset;

static char *output_file;
FILE *outfile;

static uint32_t width;
static uint32_t height;

yuv_t *image;
yuv_t *image2;

/* getopt */
extern char *optarg;


struct c63_common* init_c63_enc()
{
  /* calloc() sets allocated memory to zero */
  struct c63_common *cm = calloc(1, sizeof(struct c63_common));

  cm->width = width;
  cm->height = height;

  cm->padw[Y_COMPONENT] = cm->ypw = (uint32_t)(ceil(width/16.0f)*16);
  cm->padh[Y_COMPONENT] = cm->yph = (uint32_t)(ceil(height/16.0f)*16);
  cm->padw[U_COMPONENT] = cm->upw = (uint32_t)(ceil(width*UX/(YX*8.0f))*8);
  cm->padh[U_COMPONENT] = cm->uph = (uint32_t)(ceil(height*UY/(YY*8.0f))*8);
  cm->padw[V_COMPONENT] = cm->vpw = (uint32_t)(ceil(width*VX/(YX*8.0f))*8);
  cm->padh[V_COMPONENT] = cm->vph = (uint32_t)(ceil(height*VY/(YY*8.0f))*8);

  cm->mb_cols = cm->ypw / 8;
  cm->mb_rows = cm->yph / 8;

  /* Quality parameters -- Home exam deliveries should have original values,
   i.e., quantization factor should be 25, search range should be 16, and the
   keyframe interval should be 100. */
  cm->qp = 25;                  // Constant quantization factor. Range: [1..50]
  cm->me_search_range = 16;     // Pixels in every direction
  cm->keyframe_interval = 100;  // Distance between keyframes

  /* Initialize quantization tables */
  int i;
  for (i = 0; i < 64; ++i)
  {
    cm->quanttbl[Y_COMPONENT][i] = yquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[U_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[V_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
  }

  cm->curframe = malloc(sizeof(struct frame));
  cm->curframe->residuals = malloc(sizeof(dct_t));
  return cm;
}


static sci_error_t init_SISCI() {

	// Initialisation of SISCI API
	sci_error_t error;
	SCIInitialize(SCI_NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		return error;
	}

	// Initialize descriptors
	SCIOpen(&sd, SCI_NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		return error;
	}

	SCIGetLocalNodeId(localAdapterNo, &localNodeId, SCI_NO_FLAGS, &error);
	if(error != SCI_ERR_OK) {
		return error;
	}

	// Create local interrupt descriptor(s) for communication between encoder machine and writer machine
	interruptFromEncoderNo = ENCODED_FRAME_TRANSFERRED;
	SCICreateDataInterrupt(sd, &interruptFromEncoder, localAdapterNo, &interruptFromEncoderNo, SCI_NO_CALLBACK, NULL, SCI_FLAG_FIXED_INTNO, &error);
	if (error != SCI_ERR_OK) {
		fprintf(stderr,"SCICreateInterrupt failed - Error code 0x%x\n", error);
		return error;
	}

	// Connect reader node to remote interrupt at processing machine
	do {
		SCIConnectInterrupt(sd, &interruptToEncoder, encoderNodeId, localAdapterNo, DATA_WRITTEN, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	} while (error != SCI_ERR_OK);

	return SCI_ERR_OK;

}

static inline sci_error_t receive_width_and_height() {
	printf("Waiting for width and height from encoder...\n");
	uint32_t widthAndHeight[2];
	unsigned int length = 2*sizeof(uint32_t);

	sci_error_t error;
	SCIWaitForDataInterrupt(interruptFromEncoder, &widthAndHeight, &length, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	if(error != SCI_ERR_OK) {
		return error;
	}

	width = widthAndHeight[0];
	height = widthAndHeight[1];

	printf("Done\n");
	return SCI_ERR_OK;
}

static sci_error_t init_SISCI_segments(struct c63_common *cm) {
	sci_error_t error;

	// Set local segment id
	uint32_t localSegmentId = (localNodeId << 16) | (encoderNodeId << 8) | SEGMENT_WRITER_ENCODED;

	// Set segment size
	uint32_t localSegmentSize = sizeof(int) + (cm->mb_rows * cm->mb_cols + (cm->mb_rows/2)*(cm->mb_cols/2) + (cm->mb_rows/2)*(cm->mb_cols/2))*sizeof(struct macroblock) +
			(cm->ypw*cm->yph + cm->upw*cm->uph + cm->vpw*cm->vph) * sizeof(int16_t);

	// Create the local segment for the processing machine to copy into
	SCICreateSegment(sd, &localSegment, localSegmentId, localSegmentSize, SCI_NO_CALLBACK, NULL, SCI_NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		return error;
	}

	// Map the local segments
	int offset = 0;
	local_buffer = SCIMapLocalSegment(localSegment , &localMap, offset, localSegmentSize, NULL, SCI_NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		return error;
	}

	// Make segments accessible from the network adapter
	SCIPrepareSegment(localSegment, localAdapterNo, SCI_NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		return error;
	}

	// Make segments accessible from other nodes
	SCISetSegmentAvailable(localSegment, localAdapterNo, SCI_NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		return error;
	}

	// Set offsets within segment
	keyframe_offset = 0;

	uint32_t mb_offset_Y = keyframe_offset + sizeof(int);
	uint32_t residuals_offset_Y = mb_offset_Y + cm->mb_rows*cm->mb_cols*sizeof(struct macroblock);

	uint32_t mb_offset_U = residuals_offset_Y + cm->ypw*cm->yph*sizeof(int16_t);
	uint32_t residuals_offset_U = mb_offset_U + cm->mb_rows/2*cm->mb_cols/2*sizeof(struct macroblock);

	uint32_t mb_offset_V = residuals_offset_U + cm->upw*cm->uph*sizeof(int16_t);
	uint32_t residuals_offset_V = mb_offset_V +  cm->mb_rows/2*cm->mb_cols/2*sizeof(struct macroblock);

	// Set pointers to macroblocks
	cm->curframe->mbs[Y_COMPONENT] = (struct macroblock*) (local_buffer + mb_offset_Y);
	cm->curframe->mbs[U_COMPONENT] = (struct macroblock*) (local_buffer + mb_offset_U);
	cm->curframe->mbs[V_COMPONENT] = (struct macroblock*) (local_buffer + mb_offset_V);

	// Set pointers to residuals
	cm->curframe->residuals->Ydct = (int16_t*) (local_buffer + residuals_offset_Y);
	cm->curframe->residuals->Udct = (int16_t*) (local_buffer + residuals_offset_U);
	cm->curframe->residuals->Vdct = (int16_t*) (local_buffer + residuals_offset_V);

	return SCI_ERR_OK;
}

static sci_error_t cleanup_SISCI() {
	sci_error_t error;
	SCIDisconnectInterrupt(interruptToEncoder, SCI_NO_FLAGS, &error);
	SCIRemoveDataInterrupt(interruptFromEncoder, SCI_NO_FLAGS, &error);

	SCIUnmapSegment(localMap, SCI_NO_FLAGS, &error);

	SCIRemoveSegment(localSegment, SCI_NO_FLAGS, &error);

	SCIClose(sd, SCI_NO_FLAGS, &error);
	SCITerminate();

	return SCI_ERR_OK;
}

static void print_help()
{
	printf("Usage: ./c63write [options] output_file\n");
	printf("Commandline options:\n");
	printf("  -a                             Local adapter number\n");
	printf("  -r                             Encoder node ID\n");
	printf("  -o                             Output file (.c63)\n");
	printf("\n");

  exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
  int c;

  if (argc == 1) { print_help(); }

  while ((c = getopt(argc, argv, "a:r:o:")) != -1)
  {
	  switch (c)
	  {
	  	  case 'a':
	  		  localAdapterNo = atoi(optarg);
	  		  break;
	  	  case 'r':
	  		  encoderNodeId = atoi(optarg);
	  		  break;
	  	  case 'o':
	  		  output_file = optarg;
	  		  break;
	  	  default:
	  		  print_help();
	  		  break;
	  }
  }

  /*
  if (optind >= argc)
  {
    fprintf(stderr, "Error getting program options, try --help.\n");
    exit(EXIT_FAILURE);
  }
  */

  outfile = fopen(output_file, "wb");

  if (outfile == NULL)
  {
    perror("fopen");
    exit(EXIT_FAILURE);
  }

  sci_error_t error;
  error = init_SISCI();
  if (error != SCI_ERR_OK) {
	fprintf(stderr,"init_SISCI failed - Error code 0x%x\n", error);
	exit(EXIT_FAILURE);
  }

  error = receive_width_and_height();
  if (error != SCI_ERR_OK) {
  	fprintf(stderr,"receive_width_and_height failed - Error code 0x%x\n", error);
  	exit(EXIT_FAILURE);
}

  struct c63_common *cm = init_c63_enc();
  cm->e_ctx.fp = outfile;

  init_SISCI_segments(cm);
  printf("her\n");

  /* Encode input frames */
  int numframes = 0;

  uint8_t done = 0;
  unsigned int length = 1;
  while (1)
  {
	  printf("Waiting for data from encoder...\n");
	  sci_error_t error;
	  SCIWaitForDataInterrupt(interruptFromEncoder, &done, &length, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	  if(error != SCI_ERR_OK) {
		  return error;
	  }
	  printf("Done!\n");

	  if(done) {
		  break;
	  }

	  cm->curframe->keyframe = ((int*) local_buffer)[keyframe_offset];

	  write_frame(cm);
	  ++numframes;

	  SCITriggerInterrupt(interruptToEncoder, SCI_NO_FLAGS, &error);
	  if (error != SCI_ERR_OK) {
		  fprintf(stderr,"SCITriggerInterrupt failed - Error code 0x%x\n", error);
		  exit(EXIT_FAILURE);
	  }
  }

  free(cm->curframe->residuals);
  free(cm->curframe);
  free(cm);
  fclose(outfile);

  error = cleanup_SISCI();
  if (error != SCI_ERR_OK) {
	fprintf(stderr, "Error during SISCI cleanup - error code: %x\n", error);
	exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}
