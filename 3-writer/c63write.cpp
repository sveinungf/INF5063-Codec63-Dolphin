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

static const int Y = Y_COMPONENT;
static const int U = U_COMPONENT;
static const int V = V_COMPONENT;

struct c63_common *cms[NUM_IMAGE_SEGMENTS];
static volatile uint8_t *local_buffers[NUM_IMAGE_SEGMENTS];
static uint32_t keyframe_offset;

static char *output_file;
static FILE *outfile;

static uint32_t width;
static uint32_t height;

static pthread_mutex_t flush_mut = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t flush_cond = PTHREAD_COND_INITIALIZER;
static int thread_done = 0;
static int fd;

/* getopt */
extern char *optarg;

struct c63_common* init_c63_enc()
{
	/* calloc() sets allocated memory to zero */
	struct c63_common *cm = (struct c63_common *) calloc(1, sizeof(struct c63_common));

	cm->width = width;
	cm->height = height;

	cm->padw[Y_COMPONENT] = cm->ypw = (uint32_t) (ceil(width / 16.0f) * 16);
	cm->padh[Y_COMPONENT] = cm->yph = (uint32_t) (ceil(height / 16.0f) * 16);
	cm->padw[U_COMPONENT] = cm->upw = (uint32_t) (ceil(width * UX / (YX * 8.0f)) * 8);
	cm->padh[U_COMPONENT] = cm->uph = (uint32_t) (ceil(height * UY / (YY * 8.0f)) * 8);
	cm->padw[V_COMPONENT] = cm->vpw = (uint32_t) (ceil(width * VX / (YX * 8.0f)) * 8);
	cm->padh[V_COMPONENT] = cm->vph = (uint32_t) (ceil(height * VY / (YY * 8.0f)) * 8);

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

/*
 * Sets pointers in a c63_common struct to point into SISCI segments
 */
static void set_pointers(struct c63_common *cm, int segNum)
{
	// Set offsets within segment
	keyframe_offset = 0;

	uint32_t mb_offset_Y = keyframe_offset + sizeof(int);
	uint32_t mb_offset_U = mb_offset_Y + cm->mb_rows[Y] * cm->mb_cols[Y] * sizeof(struct macroblock);
	uint32_t mb_offset_V = mb_offset_U + cm->mb_rows[U] * cm->mb_cols[U] * sizeof(struct macroblock);

	uint32_t residuals_offset_Y = mb_offset_V + cm->mb_rows[V] * cm->mb_cols[V] * sizeof(struct macroblock);
	uint32_t residuals_offset_U = residuals_offset_Y + cm->ypw * cm->yph * sizeof(int16_t);
	uint32_t residuals_offset_V = residuals_offset_U + cm->upw * cm->uph * sizeof(int16_t);

	// Set pointers to macroblocks
	cm->curframe->mbs[Y_COMPONENT] = (struct macroblock*) (local_buffers[segNum] + mb_offset_Y);
	cm->curframe->mbs[U_COMPONENT] = (struct macroblock*) (local_buffers[segNum] + mb_offset_U);
	cm->curframe->mbs[V_COMPONENT] = (struct macroblock*) (local_buffers[segNum] + mb_offset_V);

	// Set pointers to residuals
	cm->curframe->residuals->Ydct = (int16_t*) (local_buffers[segNum] + residuals_offset_Y);
	cm->curframe->residuals->Udct = (int16_t*) (local_buffers[segNum] + residuals_offset_U);
	cm->curframe->residuals->Vdct = (int16_t*) (local_buffers[segNum] + residuals_offset_V);
}


/*
 * Function used by a dedicated thread to flush the file buffer
 */
static void *flush(void*)
{
	pthread_mutex_lock(&flush_mut);
	while (thread_done == 0)
	{
		pthread_cond_wait(&flush_cond, &flush_mut);
		fsync(fd);
	}
	pthread_mutex_unlock(&flush_mut);
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


void interrupt_handler(int)
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

	if (argc == 1)
	{
		print_help();
	}

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

	uint32_t localSegmentSize = sizeof(int);
	for (int c = 0; c < COLOR_COMPONENTS; ++c)
	{
		localSegmentSize += cms[0]->mb_cols[c] * cms[0]->mb_rows[c] * sizeof(struct macroblock);
		localSegmentSize += cms[0]->padw[c] * cms[0]->padh[c] * sizeof(int16_t);
	}

	int i;
	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
		local_buffers[i] = init_local_segment(localSegmentSize, i);
		set_pointers(cms[i], i);
	}

	/* Encode input frames */
	int numframes = 0;

	uint8_t done = 0;
	unsigned int length = sizeof(uint8_t);

	fd = fileno(outfile);
	pthread_t child;
	pthread_create(&child, NULL, flush, NULL);

	int segNum = 0;

	vector<uint8_t> byte_vector;

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
		write_frame_to_buffer(cms[segNum], byte_vector);

		// Signal encoder that writer is ready for a new frame
		signal_encoder(segNum);

		write_buffer_to_file(byte_vector, outfile);

		byte_vector.clear();


		// Flush - lock is not taken since fsync is not critical
		pthread_cond_signal(&flush_cond);

		printf(", written\n");
		++numframes;

		segNum ^= 1;
	}

	pthread_mutex_lock(&flush_mut);
	thread_done = 1;
	pthread_cond_signal(&flush_cond);
	pthread_mutex_unlock(&flush_mut);

	pthread_join(child, NULL);
	pthread_cond_destroy(&flush_cond);
	pthread_mutex_destroy(&flush_mut);

	cleanup_SISCI();

	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i)
	{
		free(cms[i]->curframe->residuals);
		free(cms[i]->curframe);
		free(cms[i]);
	}

	fclose(outfile);

	return EXIT_SUCCESS;
}
