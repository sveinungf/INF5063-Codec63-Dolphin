#include <sisci_api.h>
#include <sisci_error.h>

#include "../common/sisci_errchk.h"
#include "sisci.h"

// Local
static unsigned int localAdapterNo;
static unsigned int localNodeId;
static sci_desc_t sd;
static sci_desc_t vd;

// Reader
static unsigned int readerNodeId;
static sci_local_data_interrupt_t interruptFromReader;
static sci_remote_interrupt_t interruptToReader;
static sci_local_segment_t imageSegment;
static sci_map_t imageMap;

// Writer
static unsigned int writerNodeId;
static sci_local_interrupt_t interruptFromWriter;
static sci_remote_data_interrupt_t interruptToWriter;
static sci_remote_segment_t encodedDataSegmentWriter;

static uint32_t segmentSizeWriter;

static sci_local_segment_t encodedDataSegmentLocal;
static sci_map_t encodedDataMapLocal;
static sci_dma_queue_t dmaQueue;

uint32_t totalSize;
unsigned int keyframeSize;
unsigned int mbSizeY;
unsigned int mbSizeU;
unsigned int mbSizeV;
unsigned int residualsSizeY;
unsigned int residualsSizeU;
unsigned int residualsSizeV;

uint32_t keyframe_offset;
uint32_t mbOffsetY;
uint32_t residualsY_offset;
uint32_t mbOffsetU;
uint32_t residualsU_offset;
uint32_t mbOffsetV;
uint32_t residualsV_offset;

int *keyframe;
struct macroblock *mb_Y;
struct macroblock *mb_U;
struct macroblock *mb_V;
dct_t *residuals_Y;
dct_t *residuals_U;
dct_t *residuals_V;


void init_SISCI(unsigned int localAdapter, unsigned int readerNode, unsigned int writerNode)
{
	localAdapterNo = localAdapter;
	readerNodeId = readerNode;
	writerNodeId = writerNode;

	sci_error_t error;

	SCIInitialize(SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCIOpen(&sd, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCIOpen(&vd, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCIGetLocalNodeId(localAdapterNo, &localNodeId, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	unsigned int maxEntries = 1;
	SCICreateDMAQueue(vd, &dmaQueue, localAdapterNo, maxEntries, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	// Interrupts from the reader
	unsigned int interruptFromReaderNo = MORE_DATA_TRANSFERRED;
	SCICreateDataInterrupt(sd, &interruptFromReader, localAdapterNo, &interruptFromReaderNo, NULL,
			NULL, SCI_FLAG_FIXED_INTNO, &error);
	sisci_assert(error);

	// Interrupts from the writer
	unsigned int interruptFromWriterNo = DATA_WRITTEN;
	SCICreateInterrupt(vd, &interruptFromWriter, localAdapterNo, &interruptFromWriterNo, NULL,
			NULL, SCI_FLAG_FIXED_INTNO, &error);
	sisci_assert(error);

	// Interrupts to the reader
	printf("Connecting to interrupt on reader... ");
	fflush(stdout);
	do
	{
		SCIConnectInterrupt(sd, &interruptToReader, readerNodeId, localAdapterNo,
				READY_FOR_ORIG_TRANSFER, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	}
	while (error != SCI_ERR_OK);
	printf("Done!\n");

	// Interrupts to the writer
	printf("Connecting to interrupt on writer... ");
	fflush(stdout);
	do
	{
		SCIConnectDataInterrupt(vd, &interruptToWriter, writerNodeId, localAdapterNo,
				ENCODED_FRAME_TRANSFERRED, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	}
	while (error != SCI_ERR_OK);
	printf("Done!\n");
}

void cleanup_SISCI()
{
	sci_error_t error;

	SCIDisconnectDataInterrupt(interruptToWriter, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCIDisconnectInterrupt(interruptToReader, SCI_NO_FLAGS, &error);
	sisci_check(error);

	do {
		SCIRemoveInterrupt(interruptFromWriter, SCI_NO_FLAGS, &error);
	} while (error != SCI_ERR_OK);

	do {
		SCIRemoveDataInterrupt(interruptFromReader, SCI_NO_FLAGS, &error);
	} while (error != SCI_ERR_OK);

	SCIRemoveDMAQueue(dmaQueue, SCI_NO_FLAGS, &error);

	SCIClose(sd, SCI_NO_FLAGS, &error);
	sisci_check(error);
	SCIClose(vd, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCITerminate();
}

struct segment_yuv init_image_segment(struct c63_common* cm)
{
	struct segment_yuv image;
	unsigned int localSegmentId = (localNodeId << 16) | (readerNodeId << 8) | SEGMENT_ENCODER_IMAGE;

	unsigned int segmentSizeY = cm->ypw * cm->yph * sizeof(uint8_t);
	unsigned int segmentSizeU = cm->upw * cm->uph * sizeof(uint8_t);
	unsigned int segmentSizeV = cm->vpw * cm->vph * sizeof(uint8_t);
	unsigned int segmentSize = segmentSizeY + segmentSizeU + segmentSizeV;

	sci_error_t error;

	//SCICreateSegment(sd, &imageSegment, localSegmentId, segmentSize, SCI_NO_CALLBACK, NULL, SCI_NO_FLAGS, &error);
	SCICreateSegment(sd, &imageSegment, localSegmentId, segmentSize, SCI_NO_CALLBACK, NULL, SCI_FLAG_EMPTY, &error);
	sisci_assert(error);

	void *cudaBuffer;
	cudaMalloc(&cudaBuffer, segmentSize);

	struct cudaPointerAttributes attributes;
	cudaPointerGetAttributes(&attributes, cudaBuffer);

	SCIAttachPhysicalMemory(0, attributes.devicePointer, 0, segmentSize, imageSegment, SCI_FLAG_CUDA_BUFFER, &error);
	sisci_assert(error);

	SCIPrepareSegment(imageSegment, localAdapterNo, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	void* buffer = SCIMapLocalSegment(imageSegment, &imageMap, 0, segmentSize, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	unsigned int offset = 0;
	image.Y = (uint8_t*) buffer + offset;
	offset += segmentSizeY;
	image.U = (uint8_t*) buffer + offset;
	offset += segmentSizeU;
	image.V = (uint8_t*) buffer + offset;
	offset += segmentSizeV;

	SCISetSegmentAvailable(imageSegment, localAdapterNo, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	return image;
}

void init_remote_encoded_data_segment(struct c63_common* cm)
{
	unsigned int remoteSegmentId = (writerNodeId << 16) | (localNodeId << 8) | SEGMENT_WRITER_ENCODED;

	sci_error_t error;

	// Connect to remote segment on writer
	do {
		SCIConnectSegment(vd, &encodedDataSegmentWriter, writerNodeId, remoteSegmentId, localAdapterNo,
				SCI_NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	} while (error != SCI_ERR_OK);

	// Get segment size
	segmentSizeWriter = SCIGetRemoteSegmentSize(encodedDataSegmentWriter);

    keyframeSize = sizeof(int);
    mbSizeY = cm->mb_rowsY * cm->mb_colsY * sizeof(struct macroblock);
    mbSizeU = cm->mb_rowsUV * cm->mb_colsUV * sizeof(struct macroblock);
    mbSizeV = cm->mb_rowsUV * cm->mb_colsUV * sizeof(struct macroblock);
    residualsSizeY = cm->ypw * cm->yph * sizeof(int16_t);
    residualsSizeU = cm->upw * cm->uph * sizeof(int16_t);
    residualsSizeV = cm->vpw * cm->vph * sizeof(int16_t);

    keyframe_offset = 0;
    mbOffsetY = keyframe_offset + keyframeSize;
    mbOffsetU = mbOffsetY + mbSizeY;
    mbOffsetV = mbOffsetU + mbSizeU;
    residualsY_offset = mbOffsetV + mbSizeV;
    residualsU_offset = residualsY_offset + residualsSizeY;
    residualsV_offset = residualsU_offset + residualsSizeU;

}

void init_local_encoded_data_segment() {
	uint32_t localSegmentId = (localNodeId << 16) | (writerNodeId << 8) | 37;

	sci_error_t error;

	SCICreateSegment(vd, &encodedDataSegmentLocal, localSegmentId, segmentSizeWriter, SCI_NO_CALLBACK, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCIPrepareSegment(encodedDataSegmentLocal, localAdapterNo, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	void* buffer = SCIMapLocalSegment(encodedDataSegmentLocal, &encodedDataMapLocal, 0, segmentSizeWriter, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	keyframe = (int*) ((uint8_t*) buffer + keyframe_offset);

	mb_Y = (struct macroblock*) ((uint8_t*) buffer + mbOffsetY);
	mb_U = (struct macroblock*) ((uint8_t*) buffer + mbOffsetU);
	mb_V = (struct macroblock*) ((uint8_t*) buffer + mbOffsetV);

	residuals_Y = (dct_t*) ((uint8_t*) buffer + residualsY_offset);
	residuals_U = (dct_t*) ((uint8_t*) buffer + residualsU_offset);
	residuals_V = (dct_t*) ((uint8_t*) buffer + residualsV_offset);
}

static void cleanup_local_segment(sci_local_segment_t* segment, sci_map_t* map)
{
	sci_error_t error;

	SCISetSegmentUnavailable(*segment, localAdapterNo, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCIUnmapSegment(*map, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCIRemoveSegment(*segment, SCI_NO_FLAGS, &error);
	sisci_check(error);
}

static void cleanup_remote_segment(sci_remote_segment_t* segment)
{
	sci_error_t error;

	SCIDisconnectSegment(*segment, SCI_NO_FLAGS, &error);
	sisci_check(error);
}

void cleanup_segments()
{
	cleanup_local_segment(&imageSegment, &imageMap);
	cleanup_local_segment(&encodedDataSegmentLocal, &encodedDataMapLocal);
	cleanup_remote_segment(&encodedDataSegmentWriter);
}

void receive_width_and_height(uint32_t* width, uint32_t* height)
{
	sci_error_t error;

	printf("Waiting for width and height from reader... ");
	fflush(stdout);

	uint32_t widthAndHeight[2];
	unsigned int length = 2 * sizeof(uint32_t);
	SCIWaitForDataInterrupt(interruptFromReader, &widthAndHeight, &length, SCI_INFINITE_TIMEOUT,
			SCI_NO_FLAGS, &error);
	sisci_assert(error);

	*width = widthAndHeight[0];
	*height = widthAndHeight[1];
	printf("Done!\n");
}

void send_width_and_height(uint32_t width, uint32_t height) {
	sci_error_t error;

	uint32_t widthAndHeight[2] = {width, height};
	SCITriggerDataInterrupt(interruptToWriter, (void*) &widthAndHeight, 2*sizeof(uint32_t), SCI_NO_FLAGS, &error);
	sisci_assert(error);
}

int wait_for_reader()
{
	sci_error_t error;

	static unsigned int done_size = sizeof(uint8_t);
	uint8_t done;

	SCIWaitForDataInterrupt(interruptFromReader, &done, &done_size, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	return done;
}

void wait_for_writer()
{
	sci_error_t error;

	do {
		SCIWaitForInterrupt(interruptFromWriter, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	} while (error != SCI_ERR_OK);
}

void transfer_encoded_data(int keyframe_val, struct macroblock** mbs, dct_t* residuals)
{
	sci_error_t error;

	*keyframe = keyframe_val;
	memcpy(mb_Y, mbs[Y_COMPONENT], mbSizeY);
	memcpy(mb_U, mbs[U_COMPONENT], mbSizeU);
	memcpy(mb_V, mbs[V_COMPONENT], mbSizeV);

	memcpy(residuals_Y, residuals->base, residualsSizeY + residualsSizeU + residualsSizeV);
	//memcpy(residuals_U, residuals->Udct, residualsSizeU);
	//memcpy(residuals_V, residuals->Vdct, residualsSizeV);

	SCIStartDmaTransfer(dmaQueue, encodedDataSegmentLocal, encodedDataSegmentWriter, 0, segmentSizeWriter, 0, NULL, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCIWaitForDMAQueue(dmaQueue, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	sisci_assert(error);
}

void signal_reader()
{
	sci_error_t error;
	SCITriggerInterrupt(interruptToReader, SCI_NO_FLAGS, &error);
	sisci_assert(error);
}

void signal_writer(writer_signal signal)
{
	sci_error_t error;

	uint8_t data;

	switch (signal) {
		case ENCODING_FINISHED:
			data = 1;
			break;
		case DATA_TRANSFERRED:
			data = 0;
			break;
	}

	SCITriggerDataInterrupt(interruptToWriter, (void*) &data, sizeof(uint8_t), SCI_NO_FLAGS, &error);
	sisci_assert(error);
}
