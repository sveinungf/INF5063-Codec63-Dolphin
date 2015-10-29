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

static sci_desc_t vd;

static unsigned int localAdapterNo;
static unsigned int localNodeId;
static unsigned int remoteNodeId;

static unsigned int remoteSegmentId_Y;
static unsigned int remoteSegmentId_U;
static unsigned int remoteSegmentId_V;

static sci_remote_segment_t remoteSegment_Y;
static sci_remote_segment_t remoteSegment_U;
static sci_remote_segment_t remoteSegment_V;

static unsigned int segmentSize_Y;
static unsigned int segmentSize_U;
static unsigned int segmentSize_V;

static volatile uint8_t *remote_Y;
static volatile uint8_t *remote_U;
static volatile uint8_t *remote_V;

static sci_map_t remoteMap_Y;
static sci_map_t remoteMap_U;
static sci_map_t remoteMap_V;

sci_sequence_t sequence_Y;
sci_sequence_t sequence_U;
sci_sequence_t sequence_V;

sci_local_interrupt_t local_interrupt_data;
sci_remote_interrupt_t remote_interrupt_data;
sci_remote_interrupt_t remote_interrupt_finished;

static char *input_file;

static int limit_numframes = 0;

yuv_t *image;
yuv_t *image2;

static uint32_t width;
static uint32_t height;
static int ypw, yph;
static int upw, uph;
static int vpw, vph;

/* getopt */
extern int optind;
extern char *optarg;

/* Read planar YUV frames with 4:2:0 chroma sub-sampling */
static int read_yuv(FILE *file) {
	size_t len = 0;

	/* Read Y. The size of Y is the same as the size of the image. The indices
	 represents the color component (0 is Y, 1 is U, and 2 is V) */
	len += fread(image->Y, 1, width * height, file);

	/* Read U. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y
	 because (height/2)*(width/2) = (height*width)/4. */
	len += fread(image->U, 1, (width * height) / 4, file);

	/* Read V. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y. */
	len += fread(image->V, 1, (width * height) / 4, file);

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

static void init_c63_enc() {
	ypw = (uint32_t) (ceil(width/16.0f)*16);
	yph = (uint32_t) (ceil(height/16.0f)*16);
	upw = (uint32_t) (ceil(width*UX /(YX*8.0f))*8);
	uph = (uint32_t) (ceil(height*UY/(YY*8.0f))*8);
	vpw = (uint32_t) (ceil(width*VX/(YX*8.0f))*8);
	vph = (uint32_t) (ceil(height*VY/(YY*8.0f))*8);

	segmentSize_Y = ypw*yph*sizeof(uint8_t);
	segmentSize_U = upw*uph*sizeof(uint8_t);
	segmentSize_V = vpw*vph*sizeof(uint8_t);

	image = malloc(sizeof(*image));
	image->Y = malloc(segmentSize_Y);
	image->U = malloc(segmentSize_U);
	image->V = malloc(segmentSize_V);
}

static sci_error_t init_SISCI() {

	// Initialization of SISCI API
	sci_error_t error;
	SCIInitialize(SCI_NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		return error;
	}

	SCIOpen(&vd, SCI_NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		return error;
	}


	SCIGetLocalNodeId(localAdapterNo, &localNodeId, SCI_NO_FLAGS, &error);
	if(error != SCI_ERR_OK) {
		return error;
	}

	remoteSegmentId_Y = (remoteNodeId << 16) | localNodeId | Y_COMPONENT;
	remoteSegmentId_U = (remoteNodeId << 16) | localNodeId | U_COMPONENT;
	remoteSegmentId_V = (remoteNodeId << 16) | localNodeId | V_COMPONENT;

	do {
		SCIConnectSegment(vd, &remoteSegment_Y, remoteNodeId, remoteSegmentId_Y, localAdapterNo,
				SCI_NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	} while (error != SCI_ERR_OK);

	do {
		SCIConnectSegment(vd, &remoteSegment_U, remoteNodeId, remoteSegmentId_U, localAdapterNo,
				SCI_NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	} while (error != SCI_ERR_OK);

	do {
		 SCIConnectSegment(vd, &remoteSegment_V, remoteNodeId, remoteSegmentId_V, localAdapterNo,
				 SCI_NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	} while (error != SCI_ERR_OK);

	int offset = 0;

	remote_Y = SCIMapRemoteSegment(remoteSegment_Y, &remoteMap_Y, offset, segmentSize_Y, NULL, SCI_NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		return error;
	}

	remote_U = SCIMapRemoteSegment(remoteSegment_U, &remoteMap_U, offset, segmentSize_U, NULL, SCI_NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
			return error;
	}

	remote_V = SCIMapRemoteSegment(remoteSegment_V, &remoteMap_V, offset, segmentSize_V, NULL, SCI_NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
			return error;
	}

	// Create sequences for data error checking
	SCICreateMapSequence(remoteMap_Y, &sequence_Y, 0, &error);
    if (error != SCI_ERR_OK) {
    	fprintf(stderr,"SCICreateMapSequence failed - Error code 0x%x\n", error);
    	return error;
	}

    SCICreateMapSequence(remoteMap_U, &sequence_U, 0, &error);
    if (error != SCI_ERR_OK) {
		fprintf(stderr,"SCICreateMapSequence failed - Error code 0x%x\n", error);
		return error;
	}

    SCICreateMapSequence(remoteMap_V, &sequence_V, 0, &error);
	if (error != SCI_ERR_OK) {
		fprintf(stderr,"SCICreateMapSequence failed - Error code 0x%x\n", error);
		return error;
	}

	// Create local interrupt descriptor(s) for communication between reader machine and processing machine
	SCICreateInterrupt(vd, &local_interrupt_data, localAdapterNo, 0, NULL, NULL, SCI_NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		fprintf(stderr,"SCICreateInterrupt failed - Error code 0x%x\n", error);
		return error;
	}

	// Connect reader node to remote interrupts at processing machine
	do {
		SCIConnectInterrupt(vd, &remote_interrupt_data, remoteNodeId, localAdapterNo, 1, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	} while (error != SCI_ERR_OK);

	do {
		SCIConnectInterrupt(vd, &remote_interrupt_finished, remoteNodeId, localAdapterNo, 2, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	} while (error != SCI_ERR_OK);

	return SCI_ERR_OK;
}

static sci_error_t cleanup_SISCI() {
	sci_error_t error;
	SCIRemoveInterrupt(local_interrupt_data, SCI_NO_FLAGS, &error);
	SCIDisconnectInterrupt(remote_interrupt_data, SCI_NO_FLAGS, &error);
	SCIDisconnectInterrupt(remote_interrupt_finished, SCI_NO_FLAGS, &error);

	SCIRemoveSequence(sequence_Y, SCI_NO_FLAGS, &error);
	SCIRemoveSequence(sequence_U, SCI_NO_FLAGS, &error);
	SCIRemoveSequence(sequence_V, SCI_NO_FLAGS, &error);

	SCIUnmapSegment(remoteMap_Y, SCI_NO_FLAGS, &error);
	SCIUnmapSegment(remoteMap_U, SCI_NO_FLAGS, &error);
	SCIUnmapSegment(remoteMap_V, SCI_NO_FLAGS, &error);

	SCIDisconnectSegment(remoteSegment_Y, SCI_NO_FLAGS, &error);
	SCIDisconnectSegment(remoteSegment_U, SCI_NO_FLAGS, &error);
	SCIDisconnectSegment(remoteSegment_V, SCI_NO_FLAGS, &error);

	SCIClose(vd, SCI_NO_FLAGS, &error);
	SCITerminate();

	return SCI_ERR_OK;
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

int main(int argc, char **argv) {
	int c;

	if (argc == 1) {
		print_help();
	}

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
			remoteNodeId = atoi(optarg);
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

	init_c63_enc();

	sci_error_t error = init_SISCI();
	if (error != SCI_ERR_OK) {
		fprintf(stderr, "Error initialising SISCI API - error code: %x\n", error);
		exit(EXIT_FAILURE);
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

	int rc;
	while (1) {
		rc = read_yuv(infile);

		if (!rc) {
			// Signal computation node that there are no more frames to be encoded
			SCITriggerInterrupt(remote_interrupt_finished, SCI_NO_FLAGS, &error);
			break;
		}

		if (numframes != 0) {
			// Wait for interrupt
			do {
				SCIWaitForInterrupt(local_interrupt_data, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
			} while (error != SCI_ERR_OK);

		}

		// Copy new frame to remote segment
		printf("Sending frame %d to computation node, ", numframes);

		int remoteOffset = 0;

		SCIMemCpy(sequence_Y, image->Y, remoteMap_Y, remoteOffset, segmentSize_Y, SCI_FLAG_ERROR_CHECK, &error);
		if(error != SCI_ERR_OK) {
			fprintf(stderr,"SCIMemCpy failed on Y - Error code 0x%x\n", error);
			exit(EXIT_FAILURE);
		}

		SCIMemCpy(sequence_U, image->U, remoteMap_U, remoteOffset, segmentSize_U, SCI_FLAG_ERROR_CHECK, &error);
		if(error != SCI_ERR_OK) {
			fprintf(stderr,"SCIMemCpy failed on U - Error code 0x%x\n", error);
			exit(EXIT_FAILURE);
		}

		SCIMemCpy(sequence_V, image->V, remoteMap_V, remoteOffset, segmentSize_V, SCI_FLAG_ERROR_CHECK, &error);
		if(error != SCI_ERR_OK) {
			fprintf(stderr,"SCIMemCpy failed on V - Error code 0x%x\n", error);
			exit(EXIT_FAILURE);
		}

		printf("Done!\n");

		// Send interrupt to computation node signalling that the frame has been copied
		SCITriggerInterrupt(remote_interrupt_data, SCI_NO_FLAGS, &error);
		if (error != SCI_ERR_OK) {
			fprintf(stderr,"SCITriggerInterrupt failed - Error code 0x%x\n", error);
			exit(EXIT_FAILURE);
		}

		++numframes;

		if (limit_numframes && numframes >= limit_numframes) {
			// Signal computation node that there are no more frames to be encoded
			SCITriggerInterrupt(remote_interrupt_finished, SCI_NO_FLAGS, &error);
			break;
		}
	}

	free(image->Y);
	free(image->U);
	free(image->V);
	free(image);
	fclose(infile);

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
