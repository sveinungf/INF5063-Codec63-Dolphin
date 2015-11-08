#include <sisci_api.h>
#include <sisci_error.h>

#include "../common/sisci_errchk.h"
#include "sisci.h"

#define MIN_SEG_SZ 237569

// Local
static unsigned int localAdapterNo;
static unsigned int localNodeId;
static sci_desc_t reader_sds[2];
static sci_desc_t writer_sds[2];
static void *cudaBuffers[2];

// Reader
static unsigned int readerNodeId;
static sci_local_data_interrupt_t interruptFromReader;
static sci_remote_interrupt_t interruptToReader;
static sci_local_segment_t imageSegments[2];
static sci_map_t imageMaps[2];

// Writer
static uint32_t segmentSizeWriter;
static unsigned int writerNodeId;
static sci_local_interrupt_t interruptFromWriter;
static sci_remote_data_interrupt_t interruptToWriter;
static sci_remote_segment_t encodedDataSegmentsWriter[2];
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

	SCIOpen(&reader_sds[0], SCI_NO_FLAGS, &error);
	sisci_assert(error);

	int i;
	for (i = 0; i < 2; ++i) {
		SCIOpen(&writer_sds[i], SCI_NO_FLAGS, &error);
		sisci_assert(error);
	}

	SCIGetLocalNodeId(localAdapterNo, &localNodeId, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	unsigned int maxEntries = 1;
	SCICreateDMAQueue(writer_sds[0], &dmaQueue, localAdapterNo, maxEntries, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	// Interrupts from the reader
	unsigned int interruptFromReaderNo = MORE_DATA_TRANSFERRED;
	SCICreateDataInterrupt(reader_sds[0], &interruptFromReader, localAdapterNo, &interruptFromReaderNo, NULL,
			NULL, SCI_FLAG_FIXED_INTNO, &error);
	sisci_assert(error);

	// Interrupts from the writer
	unsigned int interruptFromWriterNo = DATA_WRITTEN;
	SCICreateInterrupt(writer_sds[0], &interruptFromWriter, localAdapterNo, &interruptFromWriterNo, NULL,
			NULL, SCI_FLAG_FIXED_INTNO, &error);
	sisci_assert(error);

	// Interrupts to the reader
	printf("Connecting to interrupt on reader... ");
	fflush(stdout);
	do
	{
		SCIConnectInterrupt(reader_sds[0], &interruptToReader, readerNodeId, localAdapterNo,
				READY_FOR_ORIG_TRANSFER, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	}
	while (error != SCI_ERR_OK);
	printf("Done!\n");

	// Interrupts to the writer
	printf("Connecting to interrupt on writer... ");
	fflush(stdout);
	do
	{
		SCIConnectDataInterrupt(writer_sds[0], &interruptToWriter, writerNodeId, localAdapterNo,
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

	SCIClose(reader_sds[0], SCI_NO_FLAGS, &error);
	sisci_check(error);
	SCIClose(writer_sds[0], SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCITerminate();
}


void set_sizes_offsets(struct c63_common *cm) {
    keyframeSize = sizeof(int);
    mbSizeY = cm->mb_rowsY * cm->mb_colsY * sizeof(struct macroblock);
    mbSizeU = cm->mb_rowsU * cm->mb_colsU * sizeof(struct macroblock);
    mbSizeV = cm->mb_rowsV * cm->mb_colsV * sizeof(struct macroblock);
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

struct segment_yuv init_image_segment(struct c63_common* cm, int segNum)
{
	struct segment_yuv image;
	unsigned int localSegmentId = (localNodeId << 16) | (readerNodeId << 8) | (SEGMENT_ENCODER_IMAGE + segNum);

	unsigned int imageSizeY = cm->ypw * cm->yph * sizeof(uint8_t);
	unsigned int imageSizeU = cm->upw * cm->uph * sizeof(uint8_t);
	unsigned int imageSizeV = cm->vpw * cm->vph * sizeof(uint8_t);
	unsigned int imageSize = imageSizeY + imageSizeU + imageSizeV;
	unsigned int segmentSize = imageSize;

	if(segmentSize < MIN_SEG_SZ) {
		segmentSize = MIN_SEG_SZ;
	}

	sci_error_t error;
	SCICreateSegment(reader_sds[0], &imageSegments[0], localSegmentId, segmentSize, SCI_NO_CALLBACK, NULL, SCI_FLAG_EMPTY, &error);
	sisci_assert(error);

	cudaMalloc(&cudaBuffers[0], segmentSize);

	struct cudaPointerAttributes attributes;
	cudaPointerGetAttributes(&attributes, cudaBuffers[0]);

	SCIAttachPhysicalMemory(0, attributes.devicePointer, 0, segmentSize, imageSegments[0], SCI_FLAG_CUDA_BUFFER, &error);
	sisci_assert(error);

	SCIPrepareSegment(imageSegments[0], localAdapterNo, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	void* buffer = SCIMapLocalSegment(imageSegments[0], &imageMaps[0], 0, segmentSize, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	unsigned int offset = 0;
	image.Y = (uint8_t*) buffer + offset;
	offset += imageSizeY;
	image.U = (uint8_t*) buffer + offset;
	offset += imageSizeU;
	image.V = (uint8_t*) buffer + offset;
	offset += imageSizeV;

	SCISetSegmentAvailable(imageSegments[0], localAdapterNo, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	return image;
}

void init_remote_encoded_data_segment(int segNum)
{
	unsigned int remoteSegmentId = (writerNodeId << 16) | (localNodeId << 8) | (SEGMENT_WRITER_ENCODED + segNum);

	sci_error_t error;

	// Connect to remote segment on writer
	do {
		SCIConnectSegment(writer_sds[segNum], &encodedDataSegmentsWriter[segNum], writerNodeId, remoteSegmentId, localAdapterNo,
				SCI_NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	} while (error != SCI_ERR_OK);

	// Get segment size
	segmentSizeWriter = SCIGetRemoteSegmentSize(encodedDataSegmentsWriter[segNum]);

}

void init_local_encoded_data_segment() {
	uint32_t localSegmentId = (localNodeId << 16) | (writerNodeId << 8) | 37;

	sci_error_t error;
	SCICreateSegment(writer_sds[0], &encodedDataSegmentLocal, localSegmentId, segmentSizeWriter, SCI_NO_CALLBACK, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCIPrepareSegment(encodedDataSegmentLocal, localAdapterNo, SCI_FLAG_DMA_SOURCE_ONLY, &error);
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
	cleanup_local_segment(&imageSegments[0], &imageMaps[0]);
	cleanup_local_segment(&encodedDataSegmentLocal, &encodedDataMapLocal);

	int i;
	for (i = 0; i < 2; ++i) {
		cleanup_remote_segment(&encodedDataSegmentsWriter[i]);
	}

	cudaFree(cudaBuffers[0]);
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


sci_callback_action_t dma_callback(void *arg, sci_dma_queue_t dma_queue, sci_error_t status) {
	if (status == SCI_ERR_OK) {
		// Send interrupt to computation node signaling that the frame has been transferred
		signal_writer(DATA_TRANSFERRED);
	}

	return SCI_CALLBACK_CANCEL;

}

void transfer_encoded_data(int keyframe_val, struct macroblock** mbs, dct_t* residuals, int segNum)
{
	sci_error_t error;

	*keyframe = keyframe_val;
	memcpy(mb_Y, mbs[Y_COMPONENT], mbSizeY);
	memcpy(mb_U, mbs[U_COMPONENT], mbSizeU);
	memcpy(mb_V, mbs[V_COMPONENT], mbSizeV);

	memcpy(residuals_Y, residuals->base, residualsSizeY + residualsSizeU + residualsSizeV);
	//memcpy(residuals_U, residuals->Udct, residualsSizeU);
	//memcpy(residuals_V, residuals->Vdct, residualsSizeV);

	SCIStartDmaTransfer(dmaQueue, encodedDataSegmentLocal, encodedDataSegmentsWriter[segNum], 0, segmentSizeWriter, 0, dma_callback, NULL, SCI_FLAG_USE_CALLBACK, &error);
	sisci_assert(error);

	//SCIWaitForDMAQueue(dmaQueue, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	//sisci_assert(error);
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
