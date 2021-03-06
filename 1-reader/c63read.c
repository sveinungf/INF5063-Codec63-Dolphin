#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <sisci_api.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "c63.h"
#include "sisci.h"

static unsigned int segmentSize_Y;
static unsigned int segmentSize_U;
static unsigned int segmentSize_V;

static char *input_file;

static int limit_numframes = 0;

static uint32_t width;
static uint32_t height;

/* getopt */
extern int optind;
extern char *optarg;

/* Read planar YUV frames with 4:2:0 chroma sub-sampling */
static int read_yuv(FILE *file, struct segment_yuv image) {
	size_t len = 0;

	/* Read Y. The size of Y is the same as the size of the image. The indices
	 represents the color component (0 is Y, 1 is U, and 2 is V) */
	len += fread((void*) image.Y, 1, width * height, file);

	/* Read U. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y
	 because (height/2)*(width/2) = (height*width)/4. */
	len += fread((void*) image.U, 1, (width * height) / 4, file);

	/* Read V. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y. */
	len += fread((void*) image.V, 1, (width * height) / 4, file);

	if (ferror(file)) {
		perror("ferror");
		exit(EXIT_FAILURE);
	}

	if (feof(file)) {
		return 0;
	} else if (len != width * height * 1.5) {
		fprintf(stderr, "Reached end of file, but incorrect bytes read.\n");
		fprintf(stderr, "Wrong input? (height: %d width: %d)\n", height, width);

		return 0;
	}

	return len;
}


static void init_segment_sizes() {
	unsigned int ypw = (uint32_t) (ceil(width/16.0f)*16);
	unsigned int yph = (uint32_t) (ceil(height/16.0f)*16);
	unsigned int upw = (uint32_t) (ceil(width*UX /(YX*8.0f))*8);
	unsigned int uph = (uint32_t) (ceil(height*UY/(YY*8.0f))*8);
	unsigned int vpw = (uint32_t) (ceil(width*VX/(YX*8.0f))*8);
	unsigned int vph = (uint32_t) (ceil(height*VY/(YY*8.0f))*8);

	segmentSize_Y = ypw*yph*sizeof(uint8_t);
	segmentSize_U = upw*uph*sizeof(uint8_t);
	segmentSize_V = vpw*vph*sizeof(uint8_t);
}


static void print_help() {
	printf("Usage: ./c63enc [options] input_file\n");
	printf("Commandline options:\n");
	printf("  -h                             Height of images to compress\n");
	printf("  -w                             Width of images to compress\n");
	printf("  -a                             Local adapter number\n");
	printf("  -r                             Remote node identifier\n");
	printf(
			"  [-f]                           Limit number of frames to encode\n");
	printf("\n");

	exit(EXIT_FAILURE);
}


void interrupt_handler(int signal)
{
	SCITerminate();
	exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
	struct sigaction int_handler;
	int_handler.sa_handler = interrupt_handler;
	sigemptyset(&int_handler.sa_mask);
	int_handler.sa_flags = 0;

	sigaction(SIGINT, &int_handler, NULL);

	int c;

	if (argc == 1) {
		print_help();
	}

	unsigned int localAdapterNo = 0;
	unsigned int encoderNodeId = 0;

	while ((c = getopt(argc, argv, "h:w:f:a:r:")) != -1) {
		switch (c) {
		case 'h':
			height = atoi(optarg);
			break;
		case 'w':
			width = atoi(optarg);
			break;
		case 'f':
			limit_numframes = atoi(optarg);
			break;
		case 'a':
			localAdapterNo = atoi(optarg);
			break;
		case 'r':
			encoderNodeId = atoi(optarg);
			break;
		default:
			print_help();
			break;
		}
	}

	if (optind >= argc) {
		fprintf(stderr, "Error getting program options, try --help.\n");
		exit(EXIT_FAILURE);
	}

	init_SISCI(localAdapterNo, encoderNodeId);
	send_width_and_height(width, height);
	init_segment_sizes();

	// Initialize image segments
	struct segment_yuv images[NUM_IMAGE_SEGMENTS];
	int i;
	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
		images[i] = init_image_segment(segmentSize_Y, segmentSize_U, segmentSize_V, i);
	}

	input_file = argv[optind];

	if (limit_numframes) {
		printf("Limited to %d frames.\n", limit_numframes);
	}

	FILE *infile = fopen(input_file, "rb");

	if (infile == NULL) {
		perror("fopen");
		exit(EXIT_FAILURE);
	}

	/* Encode input frames */
	int numframes = 0;

	int segNum = 0;

	while (1) {
		wait_for_image_transfer(segNum);

		int rc = read_yuv(infile, images[segNum]);

		if (!rc) {
			// No more data
			break;
		}

		if (numframes >= NUM_IMAGE_SEGMENTS) {
			// The encoder sends an interrupt when it is ready to receive the next frame
			wait_for_encoder(segNum);
		}

		// Copy new frame to remote segment
		printf("Sending frame %d to encoder... ", numframes);
		fflush(stdout);

		/* Transfer the new image asynchronously by using DMA
		 * Notification to encoder is handled by a callback
		 */
		transfer_image_async(segNum);

		++numframes;

		segNum ^= 1;

		if (limit_numframes && numframes >= limit_numframes) {
			// No more data
			break;
		}
	}

	// Signal encoder that there are no more frames to be encoded
	wait_for_encoder(segNum^1);
	signal_encoder(NO_MORE_FRAMES, segNum);

	fclose(infile);

	cleanup_segments();
	cleanup_SISCI();

	return EXIT_SUCCESS;
}
