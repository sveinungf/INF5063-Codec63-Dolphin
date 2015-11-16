#include <sisci_api.h>
#include <sisci_error.h>

#include "../common/sisci_errchk.h"
#include "sisci.h"

#include <string.h>


// SISCI variables
static sci_desc_t sds[NUM_IMAGE_SEGMENTS];

static unsigned int localAdapterNo;
static unsigned int localNodeId;
static unsigned int encoderNodeId;

static sci_local_segment_t localSegments[NUM_IMAGE_SEGMENTS];
static sci_map_t localMaps[NUM_IMAGE_SEGMENTS];

static local_syn_t encoder_syn;
static remote_ack_t encoder_ack;


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

	SCIOpen(&encoder_syn.sd, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCIOpen(&encoder_ack.sd, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCIGetLocalNodeId(localAdapterNo, &localNodeId, SCI_NO_FLAGS, &error);
	sisci_assert(error);
}


void receive_width_and_height(uint32_t *width, uint32_t *height) {
	printf("Waiting for width and height from reader... ");
	fflush(stdout);

	sci_error_t error;
	SCIWaitForLocalSegmentEvent(encoder_syn.segment, &encoderNodeId, &localAdapterNo, SCI_INFINITE_TIMEOUT,
			SCI_NO_FLAGS, &error);

	*width = encoder_syn.msg->frameNum;
	*height = encoder_syn.msg->status;

	encoder_syn.msg->frameNum = -1;
	encoder_syn.msg->status = -1;

	printf("Done!\n");
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
	SCIPrepareSegment(localSegments[segNum], localAdapterNo, SCI_FLAG_DMA_SOURCE_ONLY, &error);
	sisci_assert(error);

	// Make segment accessible from other nodes
	SCISetSegmentAvailable(localSegments[segNum], localAdapterNo, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	return local_buffer;
}

void init_msg_segments() {
	// Syn segment encoder
	sci_error_t error;
	unsigned int segmentSize = sizeof(message_t);

	uint32_t localSegmentId = getLocalSegId(localNodeId, encoderNodeId, SEGMENT_SYN);
	SCICreateSegment(encoder_syn.sd, &encoder_syn.segment, localSegmentId, segmentSize, SCI_NO_CALLBACK, NULL,
			SCI_NO_FLAGS, &error);
	sisci_assert(error);

	encoder_syn.msg = (message_t*)SCIMapLocalSegment(encoder_syn.segment, &encoder_syn.map, 0, segmentSize, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);
	memset((void*)encoder_syn.msg, -1, sizeof(message_t));

	SCIPrepareSegment(encoder_syn.segment, localAdapterNo, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCISetSegmentAvailable(encoder_syn.segment, localAdapterNo, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	//// Ack segment encoder
	uint32_t remoteSegmentId = getRemoteSegId(localNodeId, encoderNodeId, SEGMENT_ACK);
	do {
		SCIConnectSegment(encoder_ack.sd, &encoder_ack.segment, encoderNodeId, remoteSegmentId, localAdapterNo,
				SCI_NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	} while (error != SCI_ERR_OK);

	SCIMapRemoteSegment(encoder_ack.segment, &encoder_ack.map, 0, segmentSize, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCICreateMapSequence(encoder_ack.map, &encoder_ack.sequence, SCI_NO_FLAGS, &error);
	sisci_assert(error);

}

void wait_for_encoder(uint8_t *done, int32_t frameNum)
{
	sci_error_t error;

	do {
		SCIWaitForLocalSegmentEvent(encoder_syn.segment, &encoderNodeId, &localAdapterNo, 1,
				SCI_NO_FLAGS, &error);

		if(encoder_syn.msg->status == 1) {
			*done = 1;
			return;
		}
		else if (encoder_syn.msg->frameNum >= frameNum) {
			break;
		}
	} while(error != SCI_ERR_OK);

	*done = 0;
}


void signal_encoder(int32_t frameNum)
{
	sci_error_t error;

	message_t msg;
	msg.frameNum = frameNum;

	SCIMemCpy(encoder_ack.sequence, &msg, encoder_ack.map, 0, sizeof(message_t), SCI_NO_FLAGS, &error);
	sisci_assert(error);
	//SCIFlush(encoder_ack.sequence, SCI_NO_FLAGS);
}

void cleanup_SISCI() {
	sci_error_t error;
	int i;
	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
		SCIUnmapSegment(localMaps[i], SCI_NO_FLAGS, &error);
		sisci_check(error);

		//SCIRemoveSegment(localSegments[i], SCI_NO_FLAGS, &error);
		sisci_check(error);
		printf("aspok\n");

		SCIClose(sds[i], SCI_NO_FLAGS, &error);
		sisci_check(error);
	}
	SCISetSegmentUnavailable(encoder_syn.segment, localAdapterNo, SCI_NO_FLAGS, &error);
	sisci_check(error);
	SCIUnmapSegment(encoder_syn.map, SCI_NO_FLAGS, &error);
	sisci_check(error);
	SCIRemoveSegment(encoder_syn.segment, SCI_NO_FLAGS, &error);
	sisci_check(error);
	SCIClose(encoder_syn.sd, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCIRemoveSequence(encoder_ack.sequence, SCI_NO_FLAGS, &error);
	sisci_check(error);
	SCIUnmapSegment(encoder_ack.map, SCI_NO_FLAGS, &error);
	sisci_check(error);
	SCIDisconnectSegment(encoder_ack.segment, SCI_NO_FLAGS, &error);
	sisci_check(error);
	SCIClose(encoder_ack.sd, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCITerminate();
}
