#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "c63.h"
#include "c63_write.h"
#include "common.h"
#include "me.h"
#include "tables.h"

#include "sisci_api.h"

#define SCI_NO_FLAGS 0
#define SCI_NO_CALLBACK NULL

#define READY_FOR_TRANSFER 30
#define MORE_DATA_TRANSFERED 35

#define INIT_WRITER 40


// SISCI variables
static sci_desc_t sd_Y;
static sci_desc_t sd_U;
static sci_desc_t sd_V;

static unsigned int localAdapterNo;

static unsigned int localNodeId;
static unsigned int remoteNodeId;

static unsigned int localSegmentId_Y;
static unsigned int localSegmentId_U;
static unsigned int localSegmentId_V;

static sci_local_segment_t localSegment_Y;
static sci_local_segment_t localSegment_U;
static sci_local_segment_t localSegment_V;

static unsigned int segmentSize_Y;
static unsigned int segmentSize_U;
static unsigned int segmentSize_V;

static volatile uint8_t *local_Y;
static volatile uint8_t *local_U;
static volatile uint8_t *local_V;

static sci_map_t localMap_Y;
static sci_map_t localMap_U;
static sci_map_t localMap_V;

static sci_local_data_interrupt_t local_data_interrupt;
static unsigned int localInterruptNumber;

static sci_remote_interrupt_t remote_interrupt;


static char *output_file;
FILE *outfile;

static uint32_t width;
static uint32_t height;

yuv_t *image;
yuv_t *image2;

/* getopt */
extern int optind;
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

  // Set segment sizes
  segmentSize_Y = cm->ypw*cm->yph*sizeof(uint8_t);
  segmentSize_U = cm->upw*cm->uph*sizeof(uint8_t);
  segmentSize_V = cm->vpw*cm->vph*sizeof(uint8_t);

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
	SCIOpen(&sd_Y, SCI_NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		return error;
	}

	SCIOpen(&sd_U, SCI_NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		return error;
	}

	SCIOpen(&sd_V, SCI_NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		return error;
	}

	SCIGetLocalNodeId(localAdapterNo, &localNodeId, SCI_NO_FLAGS, &error);
	if(error != SCI_ERR_OK) {
		return error;
	}

	// Create local interrupt descriptor(s) for communication between encoder machine and writer machine
	localInterruptNumber = MORE_DATA_TRANSFERED;
	SCICreateDataInterrupt(sd_Y, &local_data_interrupt, localAdapterNo, &localInterruptNumber, SCI_NO_CALLBACK, NULL, SCI_FLAG_FIXED_INTNO, &error);
	if (error != SCI_ERR_OK) {
		fprintf(stderr,"SCICreateInterrupt failed - Error code 0x%x\n", error);
		return error;
	}

	// Connect reader node to remote interrupt at processing machine
	do {
		SCIConnectInterrupt(sd_Y, &remote_interrupt, remoteNodeId, localAdapterNo, READY_FOR_TRANSFER, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	} while (error != SCI_ERR_OK);

	return SCI_ERR_OK;

}

static inline sci_error_t receive_width_and_height() {
	printf("Waiting for width and height from encoder...\n");
	uint32_t widthAndHeight[2];
	unsigned int length = 2*sizeof(uint32_t);

	sci_error_t error;
	SCIWaitForDataInterrupt(local_data_interrupt, &widthAndHeight, &length, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	if(error != SCI_ERR_OK) {
		return error;
	}

	width = widthAndHeight[0];
	height = widthAndHeight[1];

	printf("Done\n");
	return SCI_ERR_OK;
}

static sci_error_t init_SISCI_segments() {
	sci_error_t error;

	localSegmentId_Y = (localNodeId << 16) | remoteNodeId | Y_COMPONENT;
	localSegmentId_U = (localNodeId << 16) | remoteNodeId | U_COMPONENT;
	localSegmentId_V = (localNodeId << 16) | remoteNodeId | V_COMPONENT;

	// Create local segments for the processing machine to copy into
	SCICreateSegment(sd_Y, &localSegment_Y, localSegmentId_Y, segmentSize_Y, SCI_NO_CALLBACK, NULL, SCI_NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		return error;
	}

	SCICreateSegment(sd_U, &localSegment_U, localSegmentId_U, segmentSize_U, SCI_NO_CALLBACK, NULL, SCI_NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		return error;
	}

	SCICreateSegment(sd_V, &localSegment_V, localSegmentId_V, segmentSize_V, SCI_NO_CALLBACK, NULL, SCI_NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		return error;
	}

	// Map the local segments
	int offset = 0;
	local_Y = SCIMapLocalSegment(localSegment_Y , &localMap_Y, offset, segmentSize_Y, NULL, SCI_NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		return error;
	}

	local_U = SCIMapLocalSegment(localSegment_U , &localMap_U, offset, segmentSize_U, NULL, SCI_NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
			return error;
	}

	local_V = SCIMapLocalSegment(localSegment_V , &localMap_V, offset, segmentSize_V, NULL, SCI_NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
			return error;
	}

	// Make segments accessible from the network adapter
	SCIPrepareSegment(localSegment_Y, localAdapterNo, SCI_NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		return error;
	}

	SCIPrepareSegment(localSegment_U, localAdapterNo, SCI_NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		return error;
	}

	SCIPrepareSegment(localSegment_V, localAdapterNo, SCI_NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		return error;
	}

	// Make segments accessible from other nodes
	SCISetSegmentAvailable(localSegment_Y, localAdapterNo, SCI_NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		return error;
	}

	SCISetSegmentAvailable(localSegment_U, localAdapterNo, SCI_NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		return error;
	}

	SCISetSegmentAvailable(localSegment_V, localAdapterNo, SCI_NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		return error;
	}

	return SCI_ERR_OK;
}

static sci_error_t cleanup_SISCI() {
	sci_error_t error;
	SCIRemoveDataInterrupt(local_data_interrupt, SCI_NO_FLAGS, &error);
	SCIDisconnectInterrupt(remote_interrupt, SCI_NO_FLAGS, &error);

	SCIUnmapSegment(localMap_Y, SCI_NO_FLAGS, &error);
	SCIUnmapSegment(localMap_U, SCI_NO_FLAGS, &error);
	SCIUnmapSegment(localMap_V, SCI_NO_FLAGS, &error);

	SCIRemoveSegment(localSegment_Y, SCI_NO_FLAGS, &error);
	SCIRemoveSegment(localSegment_U, SCI_NO_FLAGS, &error);
	SCIRemoveSegment(localSegment_V, SCI_NO_FLAGS, &error);

	SCIClose(sd_Y, SCI_NO_FLAGS, &error);
	SCIClose(sd_U, SCI_NO_FLAGS, &error);
	SCIClose(sd_V, SCI_NO_FLAGS, &error);
	SCITerminate();

	return SCI_ERR_OK;
}


void free_c63_enc(struct c63_common* cm)
{
  destroy_frame(cm->curframe);
  free(cm);
}

static void print_help()
{
  printf("Usage: ./c63enc [options] output_file\n");
  printf("Commandline options:\n");
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
	  		  remoteNodeId = atoi(optarg);
	  		  break;
	  	  case 'o':
	  		  output_file = optarg;
	  		  break;
	  	  default:
	  		  print_help();
	  		  break;
	  }
  }

  if (optind >= argc)
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

  init_SISCI_segments();

  /* Encode input frames */
  int numframes = 0;

  uint8_t done = 0;
  int length = 1;
  while (1)
  {
	  printf("Waiting for data from encoder...\n");
	  sci_error_t error;
	  SCIWaitForDataInterrupt(local_data_interrupt, &done, &length, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	  if(error != SCI_ERR_OK) {
		  return error;
	  }
	  printf("Done!\n");

	  if(done) {
		  break;
	  }

	  //cm->curframe->residuals->Ydct = segment_Y[residuals_Ydct]
	  write_frame(cm);
	  ++numframes;

	  SCITriggerInterrupt(remote_interrupt, SCI_NO_FLAGS, &error);
	  if (error != SCI_ERR_OK) {
		  fprintf(stderr,"SCITriggerInterrupt failed - Error code 0x%x\n", error);
		  exit(EXIT_FAILURE);
	  }
  }

  free_c63_enc(cm);
  fclose(outfile);

  error = cleanup_SISCI();
  if (error != SCI_ERR_OK) {
	fprintf(stderr, "Error during SISCI cleanup - error code: %x\n", error);
	exit(EXIT_FAILURE);
  }

  //int i, j;
  //for (i = 0; i < 2; ++i)
  //{
  //  printf("int freq[] = {");
  //  for (j = 0; j < ARRAY_SIZE(frequencies[i]); ++j)
  //  {
  //    printf("%d, ", frequencies[i][j]);
  //  }
  //  printf("};\n");
  //}

  return EXIT_SUCCESS;
}
