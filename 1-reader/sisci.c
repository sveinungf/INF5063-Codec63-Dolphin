#include "sisci.h"
#include "sisci_errchk.h"


// Local
static unsigned int localAdapterNo;
static unsigned int localNodeId;
static unsigned int encoderNodeId;

static sci_desc_t sds[NUM_IMAGE_SEGMENTS];
static sci_dma_queue_t dmaQueues[NUM_IMAGE_SEGMENTS];
static sci_local_segment_t localImageSegments[NUM_IMAGE_SEGMENTS];
static sci_map_t localImageMaps[NUM_IMAGE_SEGMENTS];

static unsigned int imageSize;
static int callback_arg[NUM_IMAGE_SEGMENTS];

// Encoder
static sci_local_interrupt_t interruptsFromEncoder[NUM_IMAGE_SEGMENTS];
static sci_remote_data_interrupt_t interruptsToEncoder[NUM_IMAGE_SEGMENTS];
static sci_remote_segment_t remoteImageSegments[NUM_IMAGE_SEGMENTS];

static volatile int transfer_completed[NUM_IMAGE_SEGMENTS] = {1, 1};


/*
 * Initializes the needed descriptors, interrupts and DMA queues
 */
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

	SCIGetLocalNodeId(localAdapterNo, &localNodeId, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	unsigned int maxEntries = 1;
	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
		SCICreateDMAQueue(sds[i], &dmaQueues[i], localAdapterNo, maxEntries, SCI_NO_FLAGS, &error);
		sisci_assert(error);
	}
	unsigned int interruptFromEncoderNo;
	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
		interruptFromEncoderNo = READY_FOR_ORIG_TRANSFER + i;
		SCICreateInterrupt(sds[i], &interruptsFromEncoder[i], localAdapterNo, &interruptFromEncoderNo,
				SCI_NO_CALLBACK, NULL, SCI_FLAG_FIXED_INTNO, &error);
		sisci_assert(error);
	}

	// Interrupts to the encoder
	printf("Connecting to interrupts on encoder... \n");
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


/*
 * Cleans up the SISCI segments and maps used
 */
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
}

/*
 * Disconnects SISCI interrupts and closes descriptors
 */
void cleanup_SISCI()
{
	sci_error_t error;

	int i;
	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
		SCIDisconnectDataInterrupt(interruptsToEncoder[i], SCI_NO_FLAGS, &error);
		sisci_check(error);

		do {
			SCIRemoveInterrupt(interruptsFromEncoder[i], SCI_NO_FLAGS, &error);
		} while (error != SCI_ERR_OK);

		SCIRemoveDMAQueue(dmaQueues[i], SCI_NO_FLAGS, &error);
		sisci_check(error);

		SCIClose(sds[i], SCI_NO_FLAGS, &error);
		sisci_check(error);
	}

	SCITerminate();
}


/*
 * Initializes a SISCI segment used to contain and transfer an image segment
 */
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


/*
 * Sends the width and the height of the video file to the encoder using an interrupt
 */
void send_width_and_height(uint32_t width, uint32_t height)
{
	sci_error_t error;

	uint32_t widthAndHeight[2] = { width, height };
	SCITriggerDataInterrupt(interruptsToEncoder[0], (void*) &widthAndHeight, 2 * sizeof(uint32_t),
			SCI_NO_FLAGS, &error);
	sisci_assert(error);
}


/*
 * Signals the encoder that a new image has been transferred
 * in the segment corresponding to "segNum"
 */
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

/*
 * Waits until the encoder is done processing the data in a segment
 */
void wait_for_encoder(int segNum)
{
	sci_error_t error;

	SCIWaitForInterrupt(interruptsFromEncoder[segNum], SCI_INFINITE_TIMEOUT,
				SCI_NO_FLAGS, &error);
	sisci_assert(error);
}


/*
 * Used to wait for the completion of an image transfer in a segment
 * corresponding to "segNum"
 */
void wait_for_image_transfer(int segNum)
{
	while(!transfer_completed[segNum]);
	transfer_completed[segNum] = 0;
}


/*
 * Callback function used to signal the encoder that a new image has been sent
 */
static sci_callback_action_t dma_callback(void *arg, sci_dma_queue_t dma_queue, sci_error_t status) {
	sci_callback_action_t retVal;
	if (status == SCI_ERR_OK) {
		printf("Done!\n");

		// Send interrupt to encoder signaling that the frame has been transferred
		signal_encoder(IMAGE_TRANSFERRED, *(int*) arg);

		transfer_completed[*(int*)arg] = 1;

		retVal = SCI_CALLBACK_CONTINUE;
	}
	else {
		retVal = SCI_CALLBACK_CANCEL;
		sisci_check(status);
	}

	return retVal;
}

/*
 * Transfer image asynchronously to encoder
 * The segment, DMA queue and the interrupt to be used in the callback, is
 * decided by the "segNum" variable
 */
void transfer_image_async(int segNum)
{
	sci_error_t error;

	callback_arg[segNum] = segNum;

	SCIStartDmaTransfer(dmaQueues[segNum], localImageSegments[segNum], remoteImageSegments[segNum], 0, imageSize, 0, dma_callback, &callback_arg[segNum], SCI_FLAG_USE_CALLBACK, &error);
	sisci_assert(error);
}
