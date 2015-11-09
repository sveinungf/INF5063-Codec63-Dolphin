#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include "c63.h"
#include "c63_write.h"
#include "tables.h"

#include "sisci.h"

struct c63_common *cms[NUM_IMAGE_SEGMENTS];
static volatile uint8_t *local_buffers[NUM_IMAGE_SEGMENTS];
static uint32_t keyframe_offset;

static char *output_file;
FILE *outfile;

static uint32_t width;
static uint32_t height;

yuv_t *image;
yuv_t *image2;

pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
int thread_done = 0;
int fd;

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

static void set_offsets_and_pointers(struct c63_common *cm, int segNum) {
	// Set offsets within segment
	keyframe_offset = 0;

	uint32_t mb_offset_Y = keyframe_offset + sizeof(int);
	uint32_t mb_offset_U = mb_offset_Y + cm->mb_rows*cm->mb_cols*sizeof(struct macroblock);
	uint32_t mb_offset_V = mb_offset_U + cm->mb_rows/2*cm->mb_cols/2*sizeof(struct macroblock);

	uint32_t residuals_offset_Y = mb_offset_V +  cm->mb_rows/2*cm->mb_cols/2*sizeof(struct macroblock);
	uint32_t residuals_offset_U = residuals_offset_Y + cm->ypw*cm->yph*sizeof(int16_t);
	uint32_t residuals_offset_V = residuals_offset_U + cm->upw*cm->uph*sizeof(int16_t);


	// Set pointers to macroblocks
	cm->curframe->mbs[Y_COMPONENT] = (struct macroblock*) (local_buffers[segNum] + mb_offset_Y);
	cm->curframe->mbs[U_COMPONENT] = (struct macroblock*) (local_buffers[segNum] + mb_offset_U);
	cm->curframe->mbs[V_COMPONENT] = (struct macroblock*) (local_buffers[segNum] + mb_offset_V);

	// Set pointers to residuals
	cm->curframe->residuals->Ydct = (int16_t*) (local_buffers[segNum] + residuals_offset_Y);
	cm->curframe->residuals->Udct = (int16_t*) (local_buffers[segNum] + residuals_offset_U);
	cm->curframe->residuals->Vdct = (int16_t*) (local_buffers[segNum] + residuals_offset_V);
}

static void *flush(void *arg) {
	pthread_mutex_lock(&mut);
	while (thread_done == 0) {
		pthread_cond_wait(&cond, &mut);
		fsync(fd);
	}
	pthread_mutex_unlock(&mut);
	return NULL;
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

  unsigned int localAdapterNo = 0;
  unsigned int encoderNodeId = 0;

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

  init_SISCI(localAdapterNo, encoderNodeId);

  receive_width_and_height(&width, &height);

  cms[0] = init_c63_enc();
  cms[1] = init_c63_enc();

  cms[0]->e_ctx.fp = outfile;
  cms[1]->e_ctx.fp = cms[0]->e_ctx.fp;

  uint32_t localSegmentSize = sizeof(int) + (cms[0]->mb_rows * cms[0]->mb_cols + (cms[0]->mb_rows/2)*(cms[0]->mb_cols/2) +
		  (cms[0]->mb_rows/2)*(cms[0]->mb_cols/2))*sizeof(struct macroblock) +
		  (cms[0]->ypw*cms[0]->yph + cms[0]->upw*cms[0]->uph + cms[0]->vpw*cms[0]->vph) * sizeof(int16_t);

  local_buffers[0] = init_local_segment(localSegmentSize, 0);
  local_buffers[1] = init_local_segment(localSegmentSize, 1);

  set_offsets_and_pointers(cms[0], 0);
  set_offsets_and_pointers(cms[1], 1);

  /* Encode input frames */
  int numframes = 0;

  uint8_t done = 0;
  unsigned int length = sizeof(uint8_t);

  fd = fileno(outfile);
  pthread_t child;
  pthread_create(&child, NULL, flush, NULL);

  int segNum = 0;

  while (1)
  {
	  printf("Frame %d:", numframes);
	  fflush(stdout);

	  wait_for_encoder(&done, &length, segNum);

	  if (!done)
	  {
		  printf(" Received");
		  fflush(stdout);
	  }
	  else
	  {
		  printf("\rNo more frames from encoder\n");
		  break;
	  }

	  cms[segNum]->curframe->keyframe = ((int*) local_buffers[segNum])[keyframe_offset];

	  write_frame(cms[segNum]);

	  // Flush
	  pthread_cond_signal(&cond);

	  printf(", written\n");
	  ++numframes;

	  // Signal encoder that writer is ready for a new frame
	  signal_encoder(segNum);

	  segNum ^= 1;
  }

  pthread_mutex_lock(&mut);
  thread_done = 1;
  pthread_cond_signal(&cond);
  pthread_mutex_unlock(&mut);

  cleanup_SISCI();

  int i;
  for (i = 0; i < 2; ++i) {
	  free(cms[i]->curframe->residuals);
	  free(cms[i]->curframe);
	  free(cms[i]);
  }

  fclose(outfile);

  return EXIT_SUCCESS;
}
