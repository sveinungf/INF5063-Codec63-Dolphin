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


// SISCI variables
static sci_desc_t vd;

static unsigned int localAdapterNo;
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

static volatile uint8_t local_Y;
static volatile uint8_t local_U;
static volatile uint8_t local_V;

static sci_map_t localMap_Y;
static sci_map_t localMap_U;
static sci_map_t localMap_V;

sci_local_interrupt_t local_interrupt_data;
sci_remote_interrupt_t remote_interrupt_data;
/*
sci_local_interrupt_t local_interrupt_Y;
sci_local_interrupt_t local_interrupt_U;
sci_local_interrupt_t local_interrupt_V;
*/


static char *output_file, *input_file;
FILE *outfile;

static int limit_numframes = 0;

static uint32_t width;
static uint32_t height;

/* getopt */
extern int optind;
extern char *optarg;


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
  else { cm->curframe->keyframe = 0; }

  if (!cm->curframe->keyframe)
  {
    /* Motion Estimation */
    c63_motion_estimate(cm);

    /* Motion Compensation */
    c63_motion_compensate(cm);
  }

  /* Function dump_image(), found in common.c, can be used here to check if the
     prediction is correct */

  write_frame(cm);

  ++cm->framenum;
  ++cm->frames_since_keyframe;
}

struct c63_common* init_c63_enc(int width, int height)
{
  int i;

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
  for (i = 0; i < 64; ++i)
  {
    cm->quanttbl[Y_COMPONENT][i] = yquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[U_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[V_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
  }

  return cm;
}


static sci_error_t init_SISCI(struct c63_common *cm) {

	// Initialisation of SISCI API
	sci_error_t error;
	SCIInitialize(NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		return error;
	}

	SCIOpen(&vd, NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		return error;
	}

	SCIGetLocalNodeId(localAdapterNo, &localNodeId, NO_FLAGS, &error);
	if(error != SCI_ERROR_OK) {
		return error;
	}

	localSegmentId_Y = (localNodeId << 16) | remoteNodeId | Y_COMPONENT;
	localSegmentId_U = (localNodeId << 16) | remoteNodeId | U_COMPONENT;
	localSegmentId_V = (localNodeId << 16) | remoteNodeId | V_COMPONENT;

	// Create local segments for the processing machine to copy into
	SCICreateSegment(vd, &localSegment_Y, localSegmentId_Y, cm->ypw*cm->yph*sizeof(uint8_t), NULL, NULL, NO_FLAGS, &error);
	if (error != SCI_ERROR_OK) {
		return error;
	}

	SCICreateSegment(vd, &localSegment_U, localSegmentId_U, cm->upw*cm->uph*sizeof(uint8_t), NULL, NULL, NO_FLAGS, &error);
	if (error != SCI_ERROR_OK) {
		return error;
	}

	SCICreateSegment(vd, &localSegment_V, localSegmentId_V, cm->vpw*cm->vph*sizeof(uint8_t), NULL, NULL, NO_FLAGS, &error);
	if (error != SCI_ERROR_OK) {
		return error;
	}

	// Map the local segments
	int offset = 0;
	local_Y = SCIMapLocalSegment(localSegment_Y , &localMap_Y, offset, cm->ypw*cm->yph*sizeof(uint8_t), NULL, NO_FLAGS, &error);
	if (error != SCI_ERROR_OK) {
		return error;
	}

	local_U = SCIMapLocalSegment(localSegment_U , &localMap_U, offset, cm->upw*cm->uph*sizeof(uint8_t), NULL, NO_FLAGS, &error);
	if (error != SCI_ERROR_OK) {
			return error;
	}

	local_V = SCIMapLocalSegment(localSegment_V , &localMap_V, offset, cm->vpw*cm->vph*sizeof(uint8_t), NULL, NO_FLAGS, &error);
	if (error != SCI_ERROR_OK) {
			return error;
	}

	// Make segments accessible from the network adapter
	SCIPrepareSegment(localSegment_Y, localAdapterNo, NO_FLAGS, &error);
	if (error != SCI_ERROR_OK) {
		return error;
	}

	SCIPrepareSegment(localSegment_U, localAdapterNo, NO_FLAGS, &error);
	if (error != SCI_ERROR_OK) {
		return error;
	}

	SCIPrepareSegment(localSegment_V, localAdapterNo, NO_FLAGS, &error);
	if (error != SCI_ERROR_OK) {
		return error;
	}

	// Make segments accessible from other nodes
	SCISetSegmentAvailable(localSegment_Y, localAdapterNo, NO_FLAGS, &error);
	if (error != SCI_ERROR_OK) {
		return error;
	}

	SCISetSegmentAvailable(localSegment_U, localAdapterNo, NO_FLAGS, &error);
	if (error != SCI_ERROR_OK) {
		return error;
	}

	SCISetSegmentAvailable(localSegment_V, localAdapterNo, NO_FLAGS, &error);
	if (error != SCI_ERROR_OK) {
		return error;
	}

	// Create local interrupt descriptor(s) for communication between processing machine and writer machine
	SCICreateInterrupt(vd, &local_interrupt_data, localAdapterNo, 0, NULL, NULL, NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		fprintf(stderr,"SCICreateInterrupt failed - Error code 0x%x\n",error);
		return error;
	}

	/*
	SCICreateInterrupt(vd, &local_interrupt_Y, localAdapterNo, Y_COMPONENT, NUll, NULL, NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		fprintf(stderr,"SCICreateInterrupt failed - Error code 0x%x\n",error);
		return error;
	}

	SCICreateInterrupt(vd, &local_interrupt_Y, localAdapterNo, U_COMPONENT, NULL, NULL, NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		fprintf(stderr,"SCICreateInterrupt failed - Error code 0x%x\n",error);
		return error;
	}

	SCICreateInterrupt(vd, &local_interrupt_Y, localAdapterNo, V_COMPONENT, NULL, NULL, NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		fprintf(stderr,"SCICreateInterrupt failed - Error code 0x%x\n",error);
		return error;
	}
	*/

	/*
	do {
		SCIConnectInterrupt(vd, &remote_interrupt_Y, remoteNodeId, localAdapterNo, Y_COMPONENT, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
	} while (error != SCI_ERROR_OK);

	do {
		SCIConnectInterrupt(vd, &remote_interrupt_U, remoteNodeId, localAdapterNo, U_COMPONENT, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
	} while (error != SCI_ERROR_OK);

	do {
		SCIConnectInterrupt(vd, &remote_interrupt_V, remoteNodeId, localAdapterNo, V_COMPONENT, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
	} while (error != SCI_ERROR_OK);
	*/

	// Connect reader node to remote interrupt at processing machine
	do {
		SCIConnectInterrupt(vd, &remote_interrupt_data, remoteNodeId, localAdapterNo, 1, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
	} while (error != SCI_ERROR_OK);

	return SCI_ERROR_OK;
}

static sci_error_t cleanup_SISCI() {
	SCIRemoveInterrupt(local_interrupt_data, NO_FLAGS, &error);
	SCIRemoveInterrupt(remote_interrupt_data, NO_FLAGS, &error);

	SCIUnmapSegment(localMap_Y, NO_FLAGS, &error);
	SCIUnmapSegment(localMap_U, NO_FLAGS, &error);
	SCIUnmapSegment(localMap_V, NO_FLAGS, &error);

	SCIRemoveSegment(localSegment_Y, NO_FLAGS, &error);
	SCIRemoveSegment(localSegment_U, NO_FLAGS, &error);
	SCIRemoveSegment(localSegment_V, NO_FLAGS, &error);

	SCIClose(&vd, NO_FLAGS, &error);
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
  printf("Usage: ./c63enc [options] input_file\n");
  printf("Commandline options:\n");
  printf("  -h                             Height of images to compress\n");
  printf("  -w                             Width of images to compress\n");
  printf("  -o                             Output file (.c63)\n");
  printf("  [-f]                           Limit number of frames to encode\n");
  printf("\n");

  exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
  int c;
  yuv_t *image;

  if (argc == 1) { print_help(); }

  while ((c = getopt(argc, argv, "h:w:o:f:i:")) != -1)
  {
    switch (c)
    {
      case 'h':
        height = atoi(optarg);
        break;
      case 'w':
        width = atoi(optarg);
        break;
      case 'o':
        output_file = optarg;
        break;
      case 'f':
        limit_numframes = atoi(optarg);
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

  struct c63_common *cm = init_c63_enc(width, height);
  cm->e_ctx.fp = outfile;

  input_file = argv[optind];

  if (limit_numframes) { printf("Limited to %d frames.\n", limit_numframes); }

  FILE *infile = fopen(input_file, "rb");

  if (infile == NULL)
  {
    perror("fopen");
    exit(EXIT_FAILURE);
  }

  /* Encode input frames */
  int numframes = 0;

  while (1)
  {
    image = read_yuv(infile, cm);

    if (!image) { break; }

    printf("Encoding frame %d, ", numframes);
    c63_encode_image(cm, image);

    free(image->Y);
    free(image->U);
    free(image->V);
    free(image);

    printf("Done!\n");

    ++numframes;

    if (limit_numframes && numframes >= limit_numframes) { break; }
  }

  free_c63_enc(cm);
  fclose(outfile);
  fclose(infile);

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
