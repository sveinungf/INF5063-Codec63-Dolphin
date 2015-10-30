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

#include "sisci_api.h"

#define SCI_NO_FLAGS 0
#define SCI_NO_CALLBACK NULL

#define READY_FOR_ORIG_TRANSFER 10
#define MORE_DATA_TRANSFERED 15

#define INIT_WRITER 50

static sci_desc_t sd;

static unsigned int localAdapterNo;
static unsigned int localNodeId;
static unsigned int remoteNodeId;

static unsigned int remoteSegmentId;

static sci_local_segment_t localSegment;
static sci_remote_segment_t remoteSegment;

static sci_dma_queue_t dmaQueue;

static unsigned int segmentSize_Y;
static unsigned int segmentSize_U;
static unsigned int segmentSize_V;

static sci_map_t localMap;
static volatile uint8_t* local_Y;
static volatile uint8_t* local_U;
static volatile uint8_t* local_V;

static sci_local_interrupt_t local_data_interrupt;
static unsigned int local_interrupt_number;

static sci_remote_data_interrupt_t remote_interrupt;

static char *input_file;

static int limit_numframes = 0;

static uint32_t width;
static uint32_t height;


/* getopt */
extern int optind;
extern char *optarg;

/* Read planar YUV frames with 4:2:0 chroma sub-sampling */
static int read_yuv(FILE *file) {
	size_t len = 0;

	/* Read Y. The size of Y is the same as the size of the image. The indices
	 represents the color component (0 is Y, 1 is U, and 2 is V) */
	len += fread((void*) local_Y, 1, width * height, file);

	/* Read U. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y
	 because (height/2)*(width/2) = (height*width)/4. */
	len += fread((void*) local_U, 1, (width * height) / 4, file);

	/* Read V. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y. */
	len += fread((void*) local_V, 1, (width * height) / 4, file);

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

static sci_error_t init_SISCI() {

	// Initialization of SISCI API
	sci_error_t error;
	SCIInitialize(SCI_NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		return error;
	}

	SCIOpen(&sd, SCI_NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		return error;
	}

	unsigned int maxEntries = 1;
	SCICreateDMAQueue(sd, &dmaQueue, localAdapterNo, maxEntries, SCI_NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		fprintf(stderr, "SCICreateDMAQueue failed - Error code 0x%x\n", error);
		return error;
	}

	// Create local interrupt descriptor(s) for communication between reader machine and processing machine
    local_interrupt_number = READY_FOR_ORIG_TRANSFER;
	SCICreateInterrupt(sd, &local_data_interrupt, localAdapterNo, &local_interrupt_number, NULL, NULL, SCI_FLAG_FIXED_INTNO, &error);
	if (error != SCI_ERR_OK) {
		fprintf(stderr,"SCICreateInterrupt failed - Error code 0x%x\n", error);
		return error;
	}

	// Connect reader node to remote interrupt at processing machine
	printf("Connecting to interrupt on encoder...\n");
	do {
		SCIConnectDataInterrupt(sd, &remote_interrupt, remoteNodeId, localAdapterNo, MORE_DATA_TRANSFERED, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	} while (error != SCI_ERR_OK);

	printf("Done\n");

	// Send interrupt to computation node with the size of the components
	uint32_t sizes[2] = {width, height};
	SCITriggerDataInterrupt(remote_interrupt, (void*) &sizes, 2*sizeof(uint32_t), SCI_NO_FLAGS, &error);
	if (error != SCI_ERR_OK) {
		fprintf(stderr,"SCITriggerInterrupt failed - Error code 0x%x\n", error);
		exit(EXIT_FAILURE);
	}

	return SCI_ERR_OK;
}

sci_error_t init_SISCI_segments() {
	sci_error_t error;

	SCIGetLocalNodeId(localAdapterNo, &localNodeId, SCI_NO_FLAGS, &error);
	if(error != SCI_ERR_OK) {
		return error;
	}

	unsigned int totalSize = segmentSize_Y + segmentSize_U + segmentSize_V;
	unsigned int localSegmentId = (localNodeId << 16) | (remoteNodeId << 8) | 0;

	SCICreateSegment(sd, &localSegment, localSegmentId, totalSize, SCI_NO_CALLBACK, NULL, SCI_NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
		fprintf(stderr,"SCICreateSegment failed - Error code 0x%x\n", error);
		return error;
	}

    SCIPrepareSegment(localSegment, localAdapterNo, SCI_NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
		fprintf(stderr,"SCIPrepareSegment failed - Error code 0x%x\n", error);
		return error;
	}

	void* buffer = SCIMapLocalSegment(localSegment, &localMap, 0, totalSize, NULL, SCI_NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
		fprintf(stderr,"SCIMapLocalSegment failed - Error code 0x%x\n", error);
		return error;
	}

    unsigned int offset = 0;
    local_Y = (uint8_t*) buffer + offset;
    offset += segmentSize_Y;
    local_U = (uint8_t*) buffer + offset;
    offset += segmentSize_U;
    local_V = (uint8_t*) buffer + offset;
    offset += segmentSize_V;

	remoteSegmentId = (remoteNodeId << 16) | (localNodeId << 8) | 0;

	do {
		SCIConnectSegment(sd, &remoteSegment, remoteNodeId, remoteSegmentId, localAdapterNo,
				SCI_NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	} while (error != SCI_ERR_OK);

	return SCI_ERR_OK;
}

static sci_error_t cleanup_SISCI() {
	sci_error_t error;

	SCIDisconnectSegment(remoteSegment, SCI_NO_FLAGS, &error);

	SCIUnmapSegment(localMap, SCI_NO_FLAGS, &error);
	SCIRemoveSegment(localSegment, SCI_NO_FLAGS, &error);

	SCIDisconnectDataInterrupt(remote_interrupt, SCI_NO_FLAGS, &error);

	SCIRemoveInterrupt(local_data_interrupt, SCI_NO_FLAGS, &error);

	SCIRemoveDMAQueue(dmaQueue, SCI_NO_FLAGS, &error);

	SCIClose(sd, SCI_NO_FLAGS, &error);

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

	error = init_SISCI_segments();
	if (error != SCI_ERR_OK) {
		fprintf(stderr, "Error initialising segments - error code: %x\n", error);
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
	uint8_t done = 0;
	while (1) {
		rc = read_yuv(infile);

		if (!rc) {
			// No more data
			done = 1;
			break;
		}
		if (numframes != 0) {
			// Wait for interrupt
			do {
				SCIWaitForInterrupt(local_data_interrupt, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
			} while (error != SCI_ERR_OK);
		}

		// Copy new frame to remote segment
		printf("Sending frame %d to computation node\n", numframes);

		unsigned int totalSize = segmentSize_Y + segmentSize_U + segmentSize_V;
		SCIStartDmaTransfer(dmaQueue, localSegment, remoteSegment, 0, totalSize, 0, NULL, NULL, SCI_NO_FLAGS, &error);
		if(error != SCI_ERR_OK) {
			fprintf(stderr,"SCIStartDmaTransfer failed - Error code 0x%x\n", error);
			exit(EXIT_FAILURE);
		}

		SCIWaitForDMAQueue(dmaQueue, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	    if (error != SCI_ERR_OK) {
	        fprintf(stderr,"SCIWaitForDMAQueue failed - Error code 0x%x\n",error);
	        return error;
	    }

		printf("Done!\n");

		// Send interrupt to computation node signalling that the frame has been copied
		SCITriggerDataInterrupt(remote_interrupt, (void*) &done, sizeof(uint8_t), SCI_NO_FLAGS, &error);
		if (error != SCI_ERR_OK) {
			fprintf(stderr,"SCITriggerInterrupt failed - Error code 0x%x\n", error);
			exit(EXIT_FAILURE);
		}

		++numframes;

		if (limit_numframes && numframes >= limit_numframes) {
			// No more data
			done = 1;
			break;
		}
	}

	// Signal computation node that there are no more frames to be encoded
	SCITriggerDataInterrupt(remote_interrupt, (void*) &done, sizeof(uint8_t), SCI_NO_FLAGS, &error);

	fclose(infile);

	error = cleanup_SISCI();
	if (error != SCI_ERR_OK) {
		fprintf(stderr, "Error during SISCI cleanup - error code: %x\n", error);
		exit(EXIT_FAILURE);
	}

	return EXIT_SUCCESS;
}
