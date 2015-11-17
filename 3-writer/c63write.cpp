#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <sisci_api.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "c63.h"
#include "write.h"

extern "C" {
#include "sisci.h"
#include "tables.h"
}

using namespace std;

static volatile int next_to_write = 0;


static const int Y = Y_COMPONENT;
static const int U = U_COMPONENT;
static const int V = V_COMPONENT;

struct c63_common *cms[NUM_IMAGE_SEGMENTS];
static volatile uint8_t *local_buffers[NUM_IMAGE_SEGMENTS];
static uint32_t keyframe_offset;

static vector<uint8_t> byte_vectors[NUM_IMAGE_SEGMENTS];

static char *output_file;
static FILE *outfile;

static uint32_t width;
static uint32_t height;

static volatile int threads_writing[NUM_IMAGE_SEGMENTS] = {0, 0};

static pthread_mutex_t write_mut = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t write_cond = PTHREAD_COND_INITIALIZER;

static pthread_mutex_t thread_muts[NUM_IMAGE_SEGMENTS] = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER};
static pthread_cond_t thread_conds[NUM_IMAGE_SEGMENTS] = {PTHREAD_COND_INITIALIZER, PTHREAD_COND_INITIALIZER};

static pthread_t children[NUM_IMAGE_SEGMENTS];

static volatile int threads_done[NUM_IMAGE_SEGMENTS] = {0, 0};
static int fd;

/* getopt */
extern char *optarg;


struct c63_common* init_c63_enc()
{
  /* calloc() sets allocated memory to zero */
  struct c63_common *cm = (struct c63_common *) calloc(1, sizeof(struct c63_common));

  cm->width = width;
  cm->height = height;

  cm->padw[Y_COMPONENT] = cm->ypw = (uint32_t)(ceil(width/16.0f)*16);
  cm->padh[Y_COMPONENT] = cm->yph = (uint32_t)(ceil(height/16.0f)*16);
  cm->padw[U_COMPONENT] = cm->upw = (uint32_t)(ceil(width*UX/(YX*8.0f))*8);
  cm->padh[U_COMPONENT] = cm->uph = (uint32_t)(ceil(height*UY/(YY*8.0f))*8);
  cm->padw[V_COMPONENT] = cm->vpw = (uint32_t)(ceil(width*VX/(YX*8.0f))*8);
  cm->padh[V_COMPONENT] = cm->vph = (uint32_t)(ceil(height*VY/(YY*8.0f))*8);

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
  cm->keyframe_interval = 100;  // Distance between keyframes

  /* Initialize quantization tables */
  int i;
  for (i = 0; i < 64; ++i)
  {
    cm->quanttbl[Y_COMPONENT][i] = yquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[U_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[V_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
  }

  cm->curframe = (struct frame*) malloc(sizeof(struct frame));
  cm->curframe->residuals = (dct_t*) malloc(sizeof(dct_t));
  return cm;
}

static void set_offsets_and_pointers(struct c63_common *cm, int segNum) {
	// Set offsets within segment
	keyframe_offset = 0;

	uint32_t mb_offset_Y = keyframe_offset + sizeof(int);
	uint32_t mb_offset_U = mb_offset_Y + cm->mb_rows[Y]*cm->mb_cols[Y]*sizeof(struct macroblock);
	uint32_t mb_offset_V = mb_offset_U + cm->mb_rows[U]*cm->mb_cols[U]*sizeof(struct macroblock);

	uint32_t residuals_offset_Y = mb_offset_V +  cm->mb_rows[V]*cm->mb_cols[V]*sizeof(struct macroblock);
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

static void *segment_handler(void *arg) {
	int threadNum = *(int*)arg;
	pthread_mutex_lock(&thread_muts[threadNum]);
	while (threads_done[threadNum] == 0) {
		//clock_t start = clock();
	  	pthread_cond_wait(&thread_conds[threadNum], &thread_muts[threadNum]);
		if(threads_done[threadNum]) {
			break;
		}
		//clock_t end = clock();
		//float seconds = (float)(end - start) / CLOCKS_PER_SEC;
		//printf("encoder: %f\n", seconds*1000.0);

		threads_writing[threadNum] = 1;

		// Write to buffer
		cms[threadNum]->curframe->keyframe = ((int*) local_buffers[threadNum])[keyframe_offset];
		write_frame_to_buffer(cms[threadNum], byte_vectors[threadNum], threadNum);

		signal_encoder(threadNum);

		// Attempt to write to disk
		pthread_mutex_lock(&write_mut);
		while(next_to_write != threadNum) {
			pthread_cond_wait(&write_cond, &write_mut);
		}
		write_buffer_to_file(byte_vectors[threadNum], cms[threadNum]->e_ctx.fp);
		next_to_write = threadNum^1;

		// Signal that writing is done
		pthread_cond_signal(&write_cond);
		pthread_mutex_unlock(&write_mut);

		byte_vectors[threadNum].clear();
		threads_writing[threadNum] = 0;

		// Signal that thread is done
		pthread_cond_signal(&thread_conds[threadNum]);
	}
	pthread_mutex_unlock(&thread_muts[threadNum]);
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

  fd = fileno(outfile);

  init_SISCI(localAdapterNo, encoderNodeId);

  receive_width_and_height(&width, &height);

  int i;
  for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
	  cms[i] = init_c63_enc();
	  cms[i]->e_ctx.fp = outfile;
  }

  uint32_t localSegmentSize = sizeof(int);
  for (int c = 0; c < COLOR_COMPONENTS; ++c) {
	  localSegmentSize += cms[0]->mb_cols[c] * cms[0]->mb_rows[c] * sizeof(struct macroblock);
	  localSegmentSize += cms[0]->padw[c] * cms[0]->padh[c] * sizeof(int16_t);
  }

  int thread_args[2] = {0, 1};
  for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
	  local_buffers[i] = init_local_segment(localSegmentSize, i);
	  set_offsets_and_pointers(cms[i], i);
	  pthread_create(&children[i], NULL, segment_handler, (void*)&thread_args[i]);
  }

  /* Encode input frames */
  int32_t frameNum = 0;
  int segNum = 0;
  uint8_t done = 0;
  unsigned int length = sizeof(uint8_t);

  while (1)
  {
  	  printf("Frame %d:", frameNum);
  	  fflush(stdout);
	  fsync(fd);

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

  	  // Delegate handling of segment to appropriate thread
	  pthread_mutex_lock(&thread_muts[segNum]);
	  while(threads_writing[segNum]) {
		  pthread_cond_wait(&thread_conds[segNum], &thread_muts[segNum]);
	  }
	  pthread_cond_signal(&thread_conds[segNum]);
	  pthread_mutex_unlock(&thread_muts[segNum]);

	  ++frameNum;

	  segNum ^= 1;
	  printf(", written\n");

  }

  // Cleanup thread variables
  for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
	  while(threads_writing[i]);
	  pthread_mutex_lock(&thread_muts[i]);
	  threads_done[i] = 1;
	  pthread_cond_signal(&thread_conds[i]);
	  pthread_mutex_unlock(&thread_muts[i]);
	  pthread_join(children[i], NULL);
	  pthread_mutex_destroy(&thread_muts[i]);
	  pthread_cond_destroy(&thread_conds[i]);
  }
  pthread_mutex_destroy(&write_mut);
  pthread_cond_destroy(&write_cond);


  cleanup_SISCI();
  for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
	  free(cms[i]->curframe->residuals);
	  free(cms[i]->curframe);
	  free(cms[i]);
  }

  fclose(outfile);

  return EXIT_SUCCESS;
}
