#include "sisci.h"
#include "sisci_errchk.h"

#include <string.h>

// Local
static unsigned int localAdapterNo;
static unsigned int localNodeId;
static sci_desc_t sds[NUM_IMAGE_SEGMENTS];
static sci_dma_queue_t dmaQueues[NUM_IMAGE_SEGMENTS];
static sci_local_segment_t localImageSegments[NUM_IMAGE_SEGMENTS];
static sci_map_t localImageMaps[NUM_IMAGE_SEGMENTS];

static unsigned int imageSize;
static uint32_t callback_arg[NUM_IMAGE_SEGMENTS];

// Encoder
static unsigned int encoderNodeId;
static sci_remote_segment_t remoteImageSegments[NUM_IMAGE_SEGMENTS];

//Message protocol
static sci_desc_t syn_sd;
static sci_remote_segment_t syn_segment;
static sci_map_t syn_map;
static volatile message_t *syn_info;
static sci_sequence_t syn_sequence;

static sci_desc_t ack_sd;
static sci_local_segment_t ack_segment;
static sci_map_t ack_map;
static volatile message_t *ack_info;


void init_SISCI(unsigned int localAdapter, unsigned int encoderNode)
{
	localAdapterNo = localAdapter;
	encoderNodeId = encoderNode;

	sci_error_t error;
	SCIInitialize(SCI_NO_FLAGS, &error);
	sisci_assert(error);

	int i;
	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
		SCIOpen(&sds[i], SCI_NO_FLAGS, &error);
		sisci_assert(error);
	}

	SCIOpen(&syn_sd, SCI_NO_FLAGS, &error);
		sisci_assert(error);

	SCIOpen(&ack_sd, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCIGetLocalNodeId(localAdapterNo, &localNodeId, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	unsigned int maxEntries = 1;
	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
		SCICreateDMAQueue(sds[i], &dmaQueues[i], localAdapterNo, maxEntries, SCI_NO_FLAGS, &error);
		sisci_assert(error);
	}
}

void cleanup_SISCI()
{
	sci_error_t error;

	int i;
	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
		SCIRemoveDMAQueue(dmaQueues[i], SCI_NO_FLAGS, &error);
		sisci_check(error);

		SCIClose(sds[i], SCI_NO_FLAGS, &error);
		sisci_check(error);
	}

	SCIClose(syn_sd, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCIClose(ack_sd, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCITerminate();
}

void init_msg_segments() {
	sci_error_t error;
	unsigned int segmentSize = sizeof(message_t);

	// Syn segment
	uint32_t remoteSegmentId = getRemoteSegId(localNodeId, encoderNodeId, SEGMENT_SYN);
	do	{
		SCIConnectSegment(syn_sd, &syn_segment, encoderNodeId, remoteSegmentId, localAdapterNo,
				SCI_NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	} while (error != SCI_ERR_OK);

	syn_info = (message_t*)SCIMapRemoteSegment(syn_segment, &syn_map, 0, segmentSize, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCICreateMapSequence(syn_map, &syn_sequence, SCI_NO_FLAGS, &error);
	sisci_assert(error);


	/// Ack segment
	uint32_t localSegmentId = getLocalSegId(localNodeId, encoderNodeId, SEGMENT_ACK);
	SCICreateSegment(ack_sd, &ack_segment, localSegmentId, segmentSize, SCI_NO_CALLBACK, NULL,
			SCI_NO_FLAGS, &error);
	sisci_assert(error);

	ack_info = (message_t*)SCIMapLocalSegment(ack_segment, &ack_map, 0, segmentSize, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);
	memset((void*)ack_info, -1, sizeof(message_t));

	SCIPrepareSegment(ack_segment, localAdapterNo, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCISetSegmentAvailable(ack_segment, localAdapterNo, SCI_NO_FLAGS, &error);
	sisci_assert(error);
}

struct segment_yuv init_image_segment(unsigned int sizeY, unsigned int sizeU, unsigned int sizeV, int segNum)
{
	sci_error_t error;

	imageSize = sizeY + sizeU + sizeV;


	struct segment_yuv image;
	uint32_t localSegmentId = getLocalSegId(localNodeId, encoderNodeId, (c63_segment)(SEGMENT_READER_IMAGE + segNum));
	SCICreateSegment(sds[segNum], &localImageSegments[segNum], localSegmentId, imageSize, SCI_NO_CALLBACK, NULL,
			SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCIPrepareSegment(localImageSegments[segNum], localAdapterNo, SCI_FLAG_DMA_SOURCE_ONLY, &error);
	sisci_assert(error);

	uint8_t* buffer = SCIMapLocalSegment(localImageSegments[segNum], &localImageMaps[segNum], 0, imageSize, NULL,
			SCI_NO_FLAGS, &error);
	sisci_assert(error);

	image.Y = buffer;
	image.U = image.Y + sizeY;
	image.V = image.U + sizeU;

	uint32_t remoteSegmentId = getRemoteSegId(localNodeId, encoderNodeId,
			(c63_segment)(SEGMENT_ENCODER_IMAGE + segNum));

	do
	{
		SCIConnectSegment(sds[segNum], &remoteImageSegments[segNum], encoderNodeId, remoteSegmentId, localAdapterNo,
				SCI_NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	}
	while (error != SCI_ERR_OK);

	return image;
}

void cleanup_segments()
{
	sci_error_t error;

	int i;
	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
		SCIDisconnectSegment(remoteImageSegments[i], SCI_NO_FLAGS, &error);
		sisci_check(error);

		SCIUnmapSegment(localImageMaps[i], SCI_NO_FLAGS, &error);
		sisci_check(error);

		SCIRemoveSegment(localImageSegments[i], SCI_NO_FLAGS, &error);
		sisci_check(error);
	}

	SCIRemoveSequence(syn_sequence, SCI_NO_FLAGS, &error);
	sisci_check(error);
	SCIUnmapSegment(syn_map, SCI_NO_FLAGS, &error);
	sisci_check(error);
	SCIDisconnectSegment(syn_segment, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCIUnmapSegment(ack_map, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCIRemoveSegment(ack_segment, SCI_NO_FLAGS, &error);
	sisci_check(error);
}

void send_width_and_height(uint32_t width, uint32_t height)
{
	sci_error_t error;

	uint32_t widthAndHeight[2] = { width, height };
	SCIMemCpy(syn_sequence, &widthAndHeight, syn_map, 0, sizeof(message_t), SCI_FLAG_ERROR_CHECK, &error);
	sisci_assert(error);
}

void wait_for_encoder(int32_t frameNum, int offset)
{
	sci_error_t error;

	do {
		SCIWaitForLocalSegmentEvent(ack_segment, &encoderNodeId, &localAdapterNo, 1,
				SCI_NO_FLAGS, &error);
		if (ack_info->frameNum >= (frameNum-offset)) {
			break;
		}
	} while(error != SCI_ERR_OK);
}

sci_callback_action_t dma_callback(void *arg, sci_dma_queue_t dma_queue, sci_error_t status) {
	sci_callback_action_t retVal;
	if (status == SCI_ERR_OK) {
		printf("Done!\n");

		// Send interrupt to computation node signaling that the frame has been transferred
		signal_encoder(IMAGE_TRANSFERRED, *(int*) arg);

		retVal = SCI_CALLBACK_CONTINUE;
	}
	else {
		retVal = SCI_CALLBACK_CANCEL;
		sisci_check(status);
	}

	return retVal;
}

void transfer_image_async(int segNum, int32_t frameNum)
{
	sci_error_t error;

	callback_arg[segNum] = frameNum;

	SCIStartDmaTransfer(dmaQueues[segNum], localImageSegments[segNum], remoteImageSegments[segNum], 0, imageSize, 0, dma_callback, &callback_arg[segNum], SCI_FLAG_USE_CALLBACK, &error);
	sisci_assert(error);
}

void wait_for_image_transfer(int segNum)
{
	sci_error_t error;

	SCIWaitForDMAQueue(dmaQueues[segNum], SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	sisci_assert(error);
}


void signal_encoder(encoder_signal signal, int frameNum)
{
	sci_error_t error;

	message_t info;
	info.frameNum = frameNum;

	switch (signal) {
		case IMAGE_TRANSFERRED:
			info.status = 0;
			break;
		case NO_MORE_FRAMES:
			info.status = 1;
			break;
	}

	SCIMemCpy(syn_sequence, &info, syn_map, 0, sizeof(message_t), SCI_FLAG_ERROR_CHECK, &error);
	sisci_assert(error);
}
