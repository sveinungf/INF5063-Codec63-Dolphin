#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <unistd.h>
//#include <pthread.h>
#include "c63.h"
#include "c63_write.h"
#include "tables.h"

#include "sisci.h"

static volatile uint8_t *local_buffer;
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

static void set_offsets_and_pointers(struct c63_common *cm) {
	// Set offsets within segment
	keyframe_offset = 0;

	uint32_t mb_offset_Y = keyframe_offset + sizeof(int);
	uint32_t mb_offset_U = mb_offset_Y + cm->mb_rows*cm->mb_cols*sizeof(struct macroblock);
	uint32_t mb_offset_V = mb_offset_U + cm->mb_rows/2*cm->mb_cols/2*sizeof(struct macroblock);

	uint32_t residuals_offset_Y = mb_offset_V +  cm->mb_rows/2*cm->mb_cols/2*sizeof(struct macroblock);
	uint32_t residuals_offset_U = residuals_offset_Y + cm->ypw*cm->yph*sizeof(int16_t);
	uint32_t residuals_offset_V = residuals_offset_U + cm->upw*cm->uph*sizeof(int16_t);


	// Set pointers to macroblocks
	cm->curframe->mbs[Y_COMPONENT] = (struct macroblock*) (local_buffer + mb_offset_Y);
	cm->curframe->mbs[U_COMPONENT] = (struct macroblock*) (local_buffer + mb_offset_U);
	cm->curframe->mbs[V_COMPONENT] = (struct macroblock*) (local_buffer + mb_offset_V);

	// Set pointers to residuals
	cm->curframe->residuals->Ydct = (int16_t*) (local_buffer + residuals_offset_Y);
	cm->curframe->residuals->Udct = (int16_t*) (local_buffer + residuals_offset_U);
	cm->curframe->residuals->Vdct = (int16_t*) (local_buffer + residuals_offset_V);
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

  struct c63_common *cm = init_c63_enc();
  cm->e_ctx.fp = outfile;

  local_buffer = init_local_segment(sizeof(int) + (cm->mb_rows * cm->mb_cols + (cm->mb_rows/2)*(cm->mb_cols/2) +
		  (cm->mb_rows/2)*(cm->mb_cols/2))*sizeof(struct macroblock) +
		  (cm->ypw*cm->yph + cm->upw*cm->uph + cm->vpw*cm->vph) * sizeof(int16_t));

  set_offsets_and_pointers(cm);

  /* Encode input frames */
  int numframes = 0;

  uint8_t done = 0;
  unsigned int length = 1;

  int fd = fileno(outfile);
  int written = 0;
  //pid_t child_pid = fork();

  while (1)
  {
	 // if(child_pid != 0) {
		  printf("Frame %d:", numframes);
		  fflush(stdout);

		  wait_for_encoder(&done, &length);

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

		  cm->curframe->keyframe = ((int*) local_buffer)[keyframe_offset];

		  write_frame(cm);
		  printf(", written\n");
		  ++numframes;

		  // Signal encoder that writer is ready for a new frame
		  signal_encoder();
		  //written = 1;
	 /* }
	  else {
		  while(!written);
		  fsync(fd);
		  written = 0;
	  }*/
  }
	  /*
  if(child_pid == 0) {
	  exit(1);
  }*/

  cleanup_SISCI();

  free(cm->curframe->residuals);
  free(cm->curframe);
  free(cm);
  fclose(outfile);

  return EXIT_SUCCESS;
}
