#include <sisci_api.h>
#include <sisci_error.h>

#include "../common/sisci_errchk.h"
#include "sisci.h"

// Local
static unsigned int localAdapterNo;
static unsigned int localNodeId;
static sci_desc_t sd;
static sci_dma_queue_t dmaQueue;
static sci_local_segment_t localImageSegment;
static sci_map_t localImageMap;

static unsigned int imageSize;

// Encoder
static unsigned int encoderNodeId;
static sci_local_interrupt_t interruptFromEncoder;
static sci_remote_data_interrupt_t interruptToEncoder;
static sci_remote_segment_t remoteImageSegment;

void init_SISCI(unsigned int localAdapter, unsigned int encoderNode)
{
	localAdapterNo = localAdapter;
	encoderNodeId = encoderNode;

	sci_error_t error;

	SCIInitialize(SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCIOpen(&sd, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCIGetLocalNodeId(localAdapterNo, &localNodeId, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	unsigned int maxEntries = 1;
	SCICreateDMAQueue(sd, &dmaQueue, localAdapterNo, maxEntries, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	// Interrupts from the encoder
	unsigned int interruptFromEncoderNo = READY_FOR_ORIG_TRANSFER;
	SCICreateInterrupt(sd, &interruptFromEncoder, localAdapterNo, &interruptFromEncoderNo,
			SCI_NO_CALLBACK, NULL, SCI_FLAG_FIXED_INTNO, &error);
	sisci_assert(error);

	// Interrupts to the encoder
	printf("Connecting to interrupt on encoder... ");
	fflush(stdout);
	do
	{
		SCIConnectDataInterrupt(sd, &interruptToEncoder, encoderNodeId, localAdapterNo,
				MORE_DATA_TRANSFERRED, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	}
	while (error != SCI_ERR_OK);
	printf("Done!\n");
}

void cleanup_SISCI()
{
	sci_error_t error;

	SCIDisconnectDataInterrupt(interruptToEncoder, SCI_NO_FLAGS, &error);
	sisci_check(error);

	do
	{
		SCIRemoveInterrupt(interruptFromEncoder, SCI_NO_FLAGS, &error);
	}
	while (error != SCI_ERR_OK);

	SCIRemoveDMAQueue(dmaQueue, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCIClose(sd, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCITerminate();
}

struct segment_yuv init_image_segment(unsigned int sizeY, unsigned int sizeU, unsigned int sizeV)
{
	struct segment_yuv image;
	unsigned int localSegmentId = (localNodeId << 16) | (encoderNodeId << 8) | 0;
	unsigned int remoteSegmentId = (encoderNodeId << 16) | (localNodeId << 8)
			| SEGMENT_ENCODER_IMAGE;

	imageSize = sizeY + sizeU + sizeV;

	sci_error_t error;

	SCICreateSegment(sd, &localImageSegment, localSegmentId, imageSize, SCI_NO_CALLBACK, NULL,
			SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCIPrepareSegment(localImageSegment, localAdapterNo, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	uint8_t* buffer = SCIMapLocalSegment(localImageSegment, &localImageMap, 0, imageSize, NULL,
			SCI_NO_FLAGS, &error);
	sisci_assert(error);

	image.Y = buffer;
	image.U = image.Y + sizeY;
	image.V = image.U + sizeU;

	do
	{
		SCIConnectSegment(sd, &remoteImageSegment, encoderNodeId, remoteSegmentId, localAdapterNo,
				SCI_NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	}
	while (error != SCI_ERR_OK);

	return image;
}

void cleanup_segments()
{
	sci_error_t error;

	SCIDisconnectSegment(remoteImageSegment, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCIUnmapSegment(localImageMap, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCIRemoveSegment(localImageSegment, SCI_NO_FLAGS, &error);
	sisci_check(error);
}

void send_width_and_height(uint32_t width, uint32_t height)
{
	sci_error_t error;

	uint32_t widthAndHeight[2] = { width, height };
	SCITriggerDataInterrupt(interruptToEncoder, (void*) &widthAndHeight, 2 * sizeof(uint32_t),
			SCI_NO_FLAGS, &error);
	sisci_assert(error);
}

void wait_for_encoder()
{
	sci_error_t error;

	do
	{
		SCIWaitForInterrupt(interruptFromEncoder, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	}
	while (error != SCI_ERR_OK);
}

void transfer_image_async()
{
	sci_error_t error;

	SCIStartDmaTransfer(dmaQueue, localImageSegment, remoteImageSegment, 0, imageSize, 0, SCI_NO_CALLBACK, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);
}

void wait_for_transfer_image()
{
	sci_error_t error;

	SCIWaitForDMAQueue(dmaQueue, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	sisci_assert(error);
}

void signal_encoder(encoder_signal signal)
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

	SCITriggerDataInterrupt(interruptToEncoder, (void*) &data, sizeof(uint8_t), SCI_NO_FLAGS, &error);
	sisci_assert(error);
}
