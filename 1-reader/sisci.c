#include "sisci.h"
#include "sisci_errchk.h"


// Local
static unsigned int localAdapterNo;
static unsigned int localNodeId;
static sci_desc_t sds[NUM_IMAGE_SEGMENTS];
static sci_dma_queue_t dmaQueues[NUM_IMAGE_SEGMENTS];
static sci_local_segment_t localImageSegments[NUM_IMAGE_SEGMENTS];
static sci_map_t localImageMaps[NUM_IMAGE_SEGMENTS];

static unsigned int imageSize;
static int callback_arg[NUM_IMAGE_SEGMENTS];

// Encoder
static unsigned int encoderNodeId;
static sci_local_data_interrupt_t interruptFromEncoder;
static sci_remote_data_interrupt_t interruptsToEncoder[NUM_IMAGE_SEGMENTS];
static sci_remote_segment_t remoteImageSegments[NUM_IMAGE_SEGMENTS];

//Message protocol
static sci_desc_t msg_sd;
static sci_local_segment_t msg_segment;
static sci_map_t msg_map;
static volatile uint8_t *msg_buffer;
static int seg_offsets[NUM_IMAGE_SEGMENTS] = {0, sizeof(uint32_t)};

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

	SCIOpen(&msg_sd, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCIGetLocalNodeId(localAdapterNo, &localNodeId, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	unsigned int maxEntries = 1;
	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
		SCICreateDMAQueue(sds[i], &dmaQueues[i], localAdapterNo, maxEntries, SCI_NO_FLAGS, &error);
		sisci_assert(error);
	}

	unsigned int interruptFromEncoderNo = READY_FOR_ORIG_TRANSFER;
	SCICreateDataInterrupt(sds[0], &interruptFromEncoder, localAdapterNo, &interruptFromEncoderNo,
			SCI_NO_CALLBACK, NULL, SCI_FLAG_FIXED_INTNO, &error);
	sisci_assert(error);


	// Interrupts to the encoder
	printf("Connecting to interrupts on encoder... ");
	fflush(stdout);

	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
		do
		{
			SCIConnectDataInterrupt(sds[i], &interruptsToEncoder[i], encoderNodeId, localAdapterNo,
					MORE_DATA_TRANSFERRED + i, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
		}
		while (error != SCI_ERR_OK);
	}
	printf("Done!\n");
}

void cleanup_SISCI()
{
	sci_error_t error;

	SCIDisconnectDataInterrupt(interruptsToEncoder[0], SCI_NO_FLAGS, &error);
	sisci_check(error);
	SCIDisconnectDataInterrupt(interruptsToEncoder[1], SCI_NO_FLAGS, &error);
	sisci_check(error);

	do
	{
		SCIRemoveDataInterrupt(interruptFromEncoder, SCI_NO_FLAGS, &error);
	} while (error != SCI_ERR_OK);


	int i;
	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
		SCIRemoveDMAQueue(dmaQueues[i], SCI_NO_FLAGS, &error);
		sisci_check(error);

		SCIClose(sds[i], SCI_NO_FLAGS, &error);
		sisci_check(error);
	}

	SCIClose(msg_sd, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCITerminate();
}

volatile uint8_t *init_msg_segment() {
	sci_error_t error;

	uint32_t localSegmentId = getLocalSegId(localNodeId, encoderNodeId, SEGMENT_MSG_PROT);
	unsigned int segmentSize = 2*sizeof(int);
	SCICreateSegment(msg_sd, &msg_segment, localSegmentId, segmentSize, SCI_NO_CALLBACK, NULL,
			SCI_NO_FLAGS, &error);
	sisci_assert(error);

	msg_buffer = (uint8_t*)SCIMapLocalSegment(msg_segment, &msg_map, 0, segmentSize, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCIPrepareSegment(msg_segment, localAdapterNo, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCISetSegmentAvailable(msg_segment, localAdapterNo, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	return msg_buffer;
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

	SCIUnmapSegment(msg_map, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCIRemoveSegment(msg_segment, SCI_NO_FLAGS, &error);
	sisci_check(error);
}

void send_width_and_height(uint32_t width, uint32_t height)
{
	sci_error_t error;

	uint32_t widthAndHeight[2] = { width, height };
	SCITriggerDataInterrupt(interruptsToEncoder[0], (void*) &widthAndHeight, 2 * sizeof(uint32_t),
			SCI_NO_FLAGS, &error);
	sisci_assert(error);
}
/*
void wait_for_encoder(int segNum)
{
	sci_error_t error;
	int ack;
	unsigned int length = sizeof(int);
	do {
		SCIWaitForDataInterrupt(interruptFromEncoder, &ack, &length, SCI_INFINITE_TIMEOUT,
				SCI_NO_FLAGS, &error);
		sisci_assert(error);
	} while (ack != segNum);
}
*/

void wait_for_encoder(int segNum)
{
	sci_error_t error;

	int *ack = (int*)(msg_buffer + seg_offsets[segNum]);

	do {
		SCIWaitForLocalSegmentEvent(msg_segment, &encoderNodeId, &localAdapterNo, 1,
				SCI_NO_FLAGS, &error);
		if (*ack == 1) {
			break;
		}
	} while(error != SCI_ERR_OK);

	*ack = 0;
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

void transfer_image_async(int segNum)
{
	sci_error_t error;

	callback_arg[segNum] = segNum;

	SCIStartDmaTransfer(dmaQueues[segNum], localImageSegments[segNum], remoteImageSegments[segNum], 0, imageSize, 0, dma_callback, &callback_arg[segNum], SCI_FLAG_USE_CALLBACK, &error);
	sisci_assert(error);
}

void wait_for_image_transfer(int segNum)
{
	sci_error_t error;

	SCIWaitForDMAQueue(dmaQueues[segNum], SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	sisci_assert(error);
}

void signal_encoder(encoder_signal signal, int segNum)
{
	sci_error_t error;

	uint8_t data;

	switch (signal) {
		case IMAGE_TRANSFERRED:
			data = 0;
			break;
		case NO_MORE_FRAMES:
			data = 1;
			break;
	}

	SCITriggerDataInterrupt(interruptsToEncoder[segNum], (void*) &data, sizeof(uint8_t), SCI_NO_FLAGS, &error);
	sisci_assert(error);
}
