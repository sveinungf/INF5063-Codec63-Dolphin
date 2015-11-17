#include <sisci_api.h>
#include <sisci_error.h>

#include "../common/sisci_errchk.h"
#include "sisci.h"


// SISCI variables
static sci_desc_t sds[NUM_IMAGE_SEGMENTS];

static unsigned int localAdapterNo;
static unsigned int localNodeId;
static unsigned int encoderNodeId;

static sci_local_segment_t localSegments[NUM_IMAGE_SEGMENTS];
static sci_map_t localMaps[NUM_IMAGE_SEGMENTS];

static sci_local_data_interrupt_t interruptsFromEncoder[NUM_IMAGE_SEGMENTS];
static sci_remote_interrupt_t interruptsToEncoder[NUM_IMAGE_SEGMENTS];


void init_SISCI(unsigned int localAdapter, unsigned int encoderNode) {
	localAdapterNo = localAdapter;
	encoderNodeId = encoderNode;

	// Initialisation of SISCI API
	sci_error_t error;
	SCIInitialize(SCI_NO_FLAGS, &error);
	sisci_assert(error);

	// Initialize descriptors
	int i;
	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
		SCIOpen(&sds[i], SCI_NO_FLAGS, &error);
		sisci_assert(error);
	}

	SCIGetLocalNodeId(localAdapterNo, &localNodeId, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	// Create local interrupt descriptor(s) for communication between encoder machine and writer machine
	unsigned int interruptFromEncoderNo;
	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
		interruptFromEncoderNo = ENCODED_FRAME_TRANSFERRED + i;
		SCICreateDataInterrupt(sds[i], &interruptsFromEncoder[i], localAdapterNo, &interruptFromEncoderNo, SCI_NO_CALLBACK, NULL, SCI_FLAG_FIXED_INTNO, &error);
		sisci_assert(error);
	}

	// Connect writer node to remote interrupt at processing machine
	printf("Connecting to interrupts on encoder... ");
	fflush(stdout);
	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
		do {
			SCIConnectInterrupt(sds[i], &interruptsToEncoder[i], encoderNodeId, localAdapterNo, DATA_WRITTEN + i,
					SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
		} while (error != SCI_ERR_OK);
	}
	printf("Done!\n");
}


void receive_width_and_height(uint32_t *width, uint32_t *height) {
	printf("Waiting for width and height from encoder...\n");
	uint32_t widthAndHeight[2];
	unsigned int length = 2*sizeof(uint32_t);

	sci_error_t error;
	SCIWaitForDataInterrupt(interruptsFromEncoder[0], &widthAndHeight, &length, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	*width = widthAndHeight[0];
	*height = widthAndHeight[1];

	printf("Done\n");
}

uint8_t *init_local_segment(uint32_t localSegmentSize, int segNum) {
	sci_error_t error;

	// Set local segment id
	uint32_t localSegmentId = getLocalSegId(localNodeId, encoderNodeId, (c63_segment)(SEGMENT_WRITER_ENCODED + segNum));

	// Create the local segment for the processing machine to copy into
	SCICreateSegment(sds[segNum], &localSegments[segNum], localSegmentId, localSegmentSize, SCI_NO_CALLBACK, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	// Map the local segment
	int offset = 0;
	uint8_t *local_buffer = SCIMapLocalSegment(localSegments[segNum] , &localMaps[segNum], offset, localSegmentSize, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	// Make segment accessible from the network adapter
	SCIPrepareSegment(localSegments[segNum], localAdapterNo, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	// Make segment accessible from other nodes
	SCISetSegmentAvailable(localSegments[segNum], localAdapterNo, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	return local_buffer;
}

void wait_for_encoder(uint8_t *done, unsigned int *length, int segNum) {
	sci_error_t error;
	SCIWaitForDataInterrupt(interruptsFromEncoder[segNum], done, length, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	sisci_assert(error);
}

void signal_encoder(int segNum) {
	sci_error_t error;

	SCITriggerInterrupt(interruptsToEncoder[segNum], SCI_NO_FLAGS, &error);
	sisci_assert(error);
}

void cleanup_SISCI() {
	sci_error_t error;

	int i;
	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
		SCIDisconnectInterrupt(interruptsToEncoder[i], SCI_NO_FLAGS, &error);
		sisci_check(error);

		SCIRemoveDataInterrupt(interruptsFromEncoder[i], SCI_NO_FLAGS, &error);
		sisci_check(error);

		SCIUnmapSegment(localMaps[i], SCI_NO_FLAGS, &error);
		sisci_check(error);

		SCIRemoveSegment(localSegments[i], SCI_NO_FLAGS, &error);
		sisci_check(error);

		SCIClose(sds[i], SCI_NO_FLAGS, &error);
		sisci_check(error);
	}

	SCITerminate();
}
