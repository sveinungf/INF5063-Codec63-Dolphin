#include <sisci_api.h>
#include <sisci_error.h>

#include "../common/sisci_errchk.h"
#include "sisci.h"


// SISCI variables
static sci_desc_t sds[2];

static unsigned int localAdapterNo;
static unsigned int localNodeId;
static unsigned int encoderNodeId;

static sci_local_segment_t localSegment;
static sci_map_t localMap;

static sci_local_data_interrupt_t interruptFromEncoder;
static sci_remote_interrupt_t interruptToEncoder;

static unsigned int interruptFromEncoderNo;


void init_SISCI(unsigned int localAdapter, unsigned int encoderNode) {
	localAdapterNo = localAdapter;
	encoderNodeId = encoderNode;

	// Initialisation of SISCI API
	sci_error_t error;
	SCIInitialize(SCI_NO_FLAGS, &error);
	sisci_assert(error);

	// Initialize descriptors
	SCIOpen(&sds[0], SCI_NO_FLAGS, &error);
	sisci_assert(error);

	// Initialize descriptors
	SCIOpen(&sds[1], SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCIGetLocalNodeId(localAdapterNo, &localNodeId, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	// Create local interrupt descriptor(s) for communication between encoder machine and writer machine
	interruptFromEncoderNo = ENCODED_FRAME_TRANSFERRED;
	SCICreateDataInterrupt(sds[0], &interruptFromEncoder, localAdapterNo, &interruptFromEncoderNo, SCI_NO_CALLBACK, NULL, SCI_FLAG_FIXED_INTNO, &error);
	sisci_assert(error);

	// Connect reader node to remote interrupt at processing machine
	do {
		SCIConnectInterrupt(sds[0], &interruptToEncoder, encoderNodeId, localAdapterNo, DATA_WRITTEN, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	} while (error != SCI_ERR_OK);
}


void receive_width_and_height(uint32_t *width, uint32_t *height) {
	printf("Waiting for width and height from encoder...\n");
	uint32_t widthAndHeight[2];
	unsigned int length = 2*sizeof(uint32_t);

	sci_error_t error;
	SCIWaitForDataInterrupt(interruptFromEncoder, &widthAndHeight, &length, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	*width = widthAndHeight[0];
	*height = widthAndHeight[1];

	printf("Done\n");
}

uint8_t *init_local_segment(uint32_t localSegmentSize, int segNum) {
	sci_error_t error;

	// Set local segment id
	uint32_t localSegmentId = (localNodeId << 16) | (encoderNodeId << 8) | (SEGMENT_WRITER_ENCODED + segNum);

	// Create the local segment for the processing machine to copy into
	SCICreateSegment(sds[segNum], &localSegment, localSegmentId, localSegmentSize, SCI_NO_CALLBACK, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	// Map the local segment
	int offset = 0;
	uint8_t *local_buffer = SCIMapLocalSegment(localSegment , &localMap, offset, localSegmentSize, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	// Make segment accessible from the network adapter
	SCIPrepareSegment(localSegment, localAdapterNo, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	// Make segment accessible from other nodes
	SCISetSegmentAvailable(localSegment, localAdapterNo, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	return local_buffer;
}

void wait_for_encoder(uint8_t *done, unsigned int *length) {
	sci_error_t error;
	SCIWaitForDataInterrupt(interruptFromEncoder, done, length, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	sisci_assert(error);
}

void signal_encoder() {
	sci_error_t error;
	SCITriggerInterrupt(interruptToEncoder, SCI_NO_FLAGS, &error);
	sisci_assert(error);
}

void cleanup_SISCI() {
	sci_error_t error;
	SCIDisconnectInterrupt(interruptToEncoder, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCIRemoveDataInterrupt(interruptFromEncoder, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCIUnmapSegment(localMap, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCIRemoveSegment(localSegment, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCIClose(sds[0], SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCIClose(sds[1], SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCITerminate();
}
