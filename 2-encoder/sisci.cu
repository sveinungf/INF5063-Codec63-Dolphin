#include <sisci_api.h>
#include <sisci_error.h>

#include "../common/sisci_errchk.h"
#include "sisci.h"

#define MIN_SEG_SZ 237569
#define LOK 2048

// Local
static unsigned int localAdapterNo;
static unsigned int localNodeId;
sci_desc_t reader_sds[NUM_IMAGE_SEGMENTS];
sci_desc_t writer_sds[NUM_IMAGE_SEGMENTS];
uint8_t* cudaBuffers[NUM_IMAGE_SEGMENTS];

// Reader
unsigned int readerNodeId;
sci_local_data_interrupt_t interruptsFromReader[NUM_IMAGE_SEGMENTS];
sci_remote_interrupt_t interruptsToReader[NUM_IMAGE_SEGMENTS];
sci_local_segment_t imageSegments[NUM_IMAGE_SEGMENTS] __attribute__((aligned(sizeof(sci_local_segment_t))));
sci_map_t imageMaps[NUM_IMAGE_SEGMENTS] __attribute__((aligned(sizeof(sci_map_t))));

// Writer
static unsigned int segmentSizeWriter;
static unsigned int writerNodeId;
static sci_local_interrupt_t interruptsFromWriter[NUM_IMAGE_SEGMENTS];
static sci_remote_data_interrupt_t interruptsToWriter[NUM_IMAGE_SEGMENTS];
static sci_remote_segment_t encodedDataSegmentsWriter[NUM_IMAGE_SEGMENTS];
static sci_local_segment_t encodedDataSegmentsLocal[NUM_IMAGE_SEGMENTS];
static sci_map_t encodedDataMapsLocal[NUM_IMAGE_SEGMENTS];
static sci_dma_queue_t dmaQueues[NUM_IMAGE_SEGMENTS];

unsigned int keyframeSize;
unsigned int mbSizeY;
unsigned int mbSizeU;
unsigned int mbSizeV;
unsigned int residualsSizeY;
unsigned int residualsSizeU;
unsigned int residualsSizeV;

unsigned int keyframe_offset;
unsigned int mbOffsetY;
unsigned int residualsY_offset;
unsigned int mbOffsetU;
unsigned int residualsU_offset;
unsigned int mbOffsetV;
unsigned int residualsV_offset;

volatile int *keyframe[NUM_IMAGE_SEGMENTS];
struct macroblock *mb_Y[NUM_IMAGE_SEGMENTS];
struct macroblock *mb_U[NUM_IMAGE_SEGMENTS];
struct macroblock *mb_V[NUM_IMAGE_SEGMENTS];


volatile struct macroblock *remote_mb_Y[NUM_IMAGE_SEGMENTS];
volatile struct macroblock *remote_mb_U[NUM_IMAGE_SEGMENTS];
volatile struct macroblock *remote_mb_V[NUM_IMAGE_SEGMENTS];
volatile uint8_t *remote_residuals_Y[NUM_IMAGE_SEGMENTS];
volatile uint8_t *remote_residuals_U[NUM_IMAGE_SEGMENTS];
volatile uint8_t *remote_residuals_V[NUM_IMAGE_SEGMENTS];
static sci_map_t encodedDataMapsRemote[NUM_IMAGE_SEGMENTS];


void init_SISCI(unsigned int localAdapter, unsigned int readerNode, unsigned int writerNode)
{
	localAdapterNo = localAdapter;
	readerNodeId = readerNode;
	writerNodeId = writerNode;

	sci_error_t error;

	SCIInitialize(SCI_NO_FLAGS, &error);
	sisci_assert(error);

	int i;
	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
		SCIOpen(&reader_sds[i], SCI_NO_FLAGS, &error);
		sisci_assert(error);

		SCIOpen(&writer_sds[i], SCI_NO_FLAGS, &error);
		sisci_assert(error);
	}

	SCIGetLocalNodeId(localAdapterNo, &localNodeId, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	unsigned int maxEntries = 1;
	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
		SCICreateDMAQueue(writer_sds[i], &dmaQueues[i], localAdapterNo, maxEntries, SCI_NO_FLAGS, &error);
		sisci_assert(error);
	}

	// Interrupts from the reader
	unsigned int interruptFromReaderNo;
	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
		interruptFromReaderNo = MORE_DATA_TRANSFERRED + i;
		SCICreateDataInterrupt(reader_sds[i], &interruptsFromReader[i], localAdapterNo, &interruptFromReaderNo, NULL,
				NULL, SCI_FLAG_FIXED_INTNO, &error);
		sisci_assert(error);
	}

	// Interrupts from the writer
	unsigned int interruptFromWriterNo;
	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
		interruptFromWriterNo = DATA_WRITTEN + i;
		SCICreateInterrupt(writer_sds[i], &interruptsFromWriter[i], localAdapterNo, &interruptFromWriterNo, NULL,
				NULL, SCI_FLAG_FIXED_INTNO, &error);
		sisci_assert(error);
	}

	// Interrupts to the reader
	printf("Connecting to interrupts on reader... ");
	fflush(stdout);
	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
		do
		{
			SCIConnectInterrupt(reader_sds[i], &interruptsToReader[i], readerNodeId, localAdapterNo,
					READY_FOR_ORIG_TRANSFER + i, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
		}
		while (error != SCI_ERR_OK);
	}
	printf("Done!\n");

	// Interrupts to the writer
	printf("Connecting to interrupt on writer... ");
	fflush(stdout);
	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
		do
		{
			SCIConnectDataInterrupt(writer_sds[i], &interruptsToWriter[i], writerNodeId, localAdapterNo,
					ENCODED_FRAME_TRANSFERRED + i, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
		}
		while (error != SCI_ERR_OK);
	}
	printf("Done!\n");
}

void cleanup_SISCI()
{
	sci_error_t error;

	int i;
	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
		SCIDisconnectDataInterrupt(interruptsToWriter[i], SCI_NO_FLAGS, &error);
		sisci_check(error);
		SCIDisconnectInterrupt(interruptsToReader[i], SCI_NO_FLAGS, &error);
		sisci_check(error);

		do {
				SCIRemoveInterrupt(interruptsFromWriter[i], SCI_NO_FLAGS, &error);
		} while (error != SCI_ERR_OK);

		do {
			SCIRemoveDataInterrupt(interruptsFromReader[i], SCI_NO_FLAGS, &error);
		} while (error != SCI_ERR_OK);

		SCIRemoveDMAQueue(dmaQueues[i], SCI_NO_FLAGS, &error);

		SCIClose(reader_sds[i], SCI_NO_FLAGS, &error);
		sisci_check(error);

		SCIClose(writer_sds[i], SCI_NO_FLAGS, &error);
		sisci_check(error);
	}
	SCITerminate();
}


void set_sizes_offsets(struct c63_common *cm) {
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

struct segment_yuv init_image_segment(struct c63_common* cm, int segNum)
{
	struct segment_yuv image;
	unsigned int localSegmentId = (localNodeId << 16) | (readerNodeId << 8) | (SEGMENT_ENCODER_IMAGE + segNum);

	unsigned int imageSizeY = cm->ypw * cm->yph * sizeof(uint8_t);
	unsigned int imageSizeU = cm->upw * cm->uph * sizeof(uint8_t);
	unsigned int imageSizeV = cm->vpw * cm->vph * sizeof(uint8_t);
	unsigned int imageSize = imageSizeY + imageSizeU + imageSizeV;
	unsigned int segmentSize = imageSize;

	//segmentSize = segmentSize/LOK * LOK + LOK;

	if(segmentSize < MIN_SEG_SZ) {
		segmentSize = MIN_SEG_SZ;
	}

	sci_error_t error;
	SCICreateSegment(reader_sds[segNum], &imageSegments[segNum], localSegmentId, segmentSize, SCI_NO_CALLBACK, NULL, SCI_FLAG_EMPTY, &error);
	sisci_assert(error);

	cudaMalloc((void**)&cudaBuffers[segNum], 3*segmentSize);

	struct cudaPointerAttributes attributes;
	cudaPointerGetAttributes(&attributes, (void*)cudaBuffers[segNum]);

	printf("addr: %ld\n", (long unsigned int) attributes.devicePointer);

	SCIAttachPhysicalMemory(0, attributes.devicePointer, 0, segmentSize, imageSegments[segNum], SCI_FLAG_CUDA_BUFFER, &error);
	sisci_assert(error);

	unsigned int offset = 0;
	volatile void* buffer;
	buffer = SCIMapLocalSegment(imageSegments[segNum], &imageMaps[segNum], offset, imageSizeY, NULL, SCI_FLAG_READONLY_MAP, &error);
	sisci_assert(error);
	image.Y = (volatile uint8_t*) buffer;

	offset += imageSizeY;
	buffer = SCIMapLocalSegment(imageSegments[segNum], &imageMaps[segNum], offset, imageSizeU, NULL, SCI_FLAG_READONLY_MAP, &error);
	sisci_assert(error);
	image.U = (volatile uint8_t*) buffer;

	offset += imageSizeU;
	buffer = SCIMapLocalSegment(imageSegments[segNum], &imageMaps[segNum], offset, imageSizeV, NULL, SCI_FLAG_READONLY_MAP, &error);
	sisci_assert(error);
	image.V = (volatile uint8_t*) buffer;


	/*
	unsigned int offset = 0;
	image.Y = (uint8_t*) buffer + offset;
	offset += imageSizeY;
	image.U = (uint8_t*) buffer + offset;
	offset += imageSizeU;
	image.V = (uint8_t*) buffer + offset;
	offset += imageSizeV;
	*/

	SCIPrepareSegment(imageSegments[segNum], localAdapterNo, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCISetSegmentAvailable(imageSegments[segNum], localAdapterNo, SCI_NO_FLAGS, &error);
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

	unsigned int offset = 0;

	printf("segnum: %d\n", segNum);
	/*
	volatile void *buffer;
	buffer = SCIMapRemoteSegment(encodedDataSegmentsWriter[segNum], &encodedDataMapsRemote[segNum],
			offset, keyframeSize, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);
	keyframe[segNum] = (volatile int*) (volatile uint8_t*) buffer;

	offset += keyframeSize;
	buffer = SCIMapRemoteSegment(encodedDataSegmentsWriter[segNum], &encodedDataMapsRemote[segNum],
				offset, mbSizeY, (void*)((uint8_t*)buffer+offset), SCI_FLAG_FIXED_MAP_ADDR, &error);
	sisci_assert(error);
	remote_mb_Y[segNum] = (struct macroblock*)(volatile uint8_t*)buffer;

	offset += mbSizeY;
	buffer = SCIMapRemoteSegment(encodedDataSegmentsWriter[segNum], &encodedDataMapsRemote[segNum],
				offset, mbSizeU, (void*)((uint8_t*)buffer+offset), SCI_FLAG_FIXED_MAP_ADDR, &error);
	sisci_assert(error);
	remote_mb_U[segNum] = (struct macroblock*)(volatile uint8_t*)buffer;

	offset += mbSizeU;
	buffer = SCIMapRemoteSegment(encodedDataSegmentsWriter[segNum], &encodedDataMapsRemote[segNum],
				offset, mbSizeV, (void*)((uint8_t*)buffer+offset), SCI_FLAG_FIXED_MAP_ADDR, &error);
	sisci_assert(error);
	remote_mb_V[segNum] = (struct macroblock*)(volatile uint8_t*)buffer;

	offset += mbSizeV;
	buffer = SCIMapRemoteSegment(encodedDataSegmentsWriter[segNum], &encodedDataMapsRemote[segNum],
				offset, residualsSizeY, (void*)((uint8_t*)buffer+offset), SCI_FLAG_FIXED_MAP_ADDR, &error);
	sisci_assert(error);
	remote_residuals_Y[segNum] = (volatile uint8_t*)buffer;

	offset += residualsSizeY;
	buffer = SCIMapRemoteSegment(encodedDataSegmentsWriter[segNum], &encodedDataMapsRemote[segNum],
				offset, residualsSizeU, (void*)((uint8_t*)buffer+offset), SCI_FLAG_FIXED_MAP_ADDR, &error);
	sisci_assert(error);
	remote_residuals_U[segNum] = (volatile uint8_t*)buffer;

	offset += residualsSizeU;
	buffer = SCIMapRemoteSegment(encodedDataSegmentsWriter[segNum], &encodedDataMapsRemote[segNum],
				offset, residualsSizeV, (void*)((uint8_t*)buffer+offset), SCI_FLAG_FIXED_MAP_ADDR, &error);
	sisci_assert(error);
	remote_residuals_V[segNum] = (volatile uint8_t*)buffer;
	*/
	/*
	volatile void *buffer;
	buffer = SCIMapRemoteSegment(encodedDataSegmentsWriter[segNum], &encodedDataMapsRemote[segNum],
			offset, keyframeSize, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);
	keyframe[segNum] = (volatile int*) (volatile uint8_t*) buffer;

	offset += keyframeSize;
	buffer = SCIMapRemoteSegment(encodedDataSegmentsWriter[segNum], &encodedDataMapsRemote[segNum],
				offset, mbSizeY+mbSizeU+mbSizeV, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);
	remote_mb_Y[segNum] = (struct macroblock*)(volatile uint8_t*)buffer;
	remote_mb_U[segNum] = (struct macroblock*)((volatile uint8_t*)buffer+mbSizeY);
	remote_mb_V[segNum] = (struct macroblock*)((volatile uint8_t*)buffer+mbSizeU);

	offset += mbSizeY+mbSizeU+mbSizeV;
	buffer = SCIMapRemoteSegment(encodedDataSegmentsWriter[segNum], &encodedDataMapsRemote[segNum],
				offset, residualsSizeY+residualsSizeU+residualsSizeV, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);
	remote_residuals_Y[segNum] = (volatile uint8_t*)buffer;
	remote_residuals_U[segNum] = ((volatile uint8_t*)buffer+residualsSizeY);
	remote_residuals_V[segNum] = ((volatile uint8_t*)buffer+residualsSizeU);
	 */

	volatile void *buffer;
	buffer = SCIMapRemoteSegment(encodedDataSegmentsWriter[segNum], &encodedDataMapsRemote[segNum],
			offset, segmentSizeWriter, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);
	keyframe[segNum] = (int*) ((uint8_t*)buffer + keyframe_offset);

	remote_mb_Y[segNum] = (struct macroblock*)((uint8_t*)buffer + mbOffsetY);
	remote_mb_U[segNum] = (struct macroblock*)((uint8_t*)buffer + mbOffsetU);
	remote_mb_V[segNum] = (struct macroblock*)((uint8_t*)buffer + mbOffsetV);

	remote_residuals_Y[segNum] = (uint8_t*)buffer + residualsY_offset;
	remote_residuals_U[segNum] = (uint8_t*)buffer + residualsU_offset;
	remote_residuals_V[segNum] = (uint8_t*)buffer + residualsV_offset;


}

void init_local_encoded_data_segment() {
	sci_error_t error;
	uint32_t localSegmentId = (localNodeId << 16) | (writerNodeId << 8) | 37;

	SCICreateSegment(writer_sds[0], &encodedDataSegmentsLocal[0], localSegmentId, segmentSizeWriter, SCI_NO_CALLBACK, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCIPrepareSegment(encodedDataSegmentsLocal[0], localAdapterNo, SCI_FLAG_DMA_SOURCE_ONLY, &error);
	sisci_assert(error);

	volatile void *buffer = SCIMapLocalSegment(encodedDataSegmentsLocal[0], &encodedDataMapsLocal[0], 0, segmentSizeWriter, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	keyframe[0] = (int*) ((uint8_t*)buffer + keyframe_offset);

	mb_Y[0] = (struct macroblock*) ((uint8_t*) buffer + mbOffsetY);
	mb_U[0] = (struct macroblock*) ((uint8_t*) buffer + mbOffsetU);
	mb_V[0] = (struct macroblock*) ((uint8_t*) buffer + mbOffsetV);
}


void init_local_encoded_data_segments() {
	sci_error_t error;
	uint32_t localSegmentId;
	void *buffer;

	unsigned int segmentSize = keyframeSize + mbSizeY + mbSizeU + mbSizeV;

	int i;
	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {

		localSegmentId = (localNodeId << 16) | (writerNodeId << 8) | (37 + i);

		SCICreateSegment(writer_sds[i], &encodedDataSegmentsLocal[i], localSegmentId, segmentSize, SCI_NO_CALLBACK, NULL, SCI_NO_FLAGS, &error);
		sisci_assert(error);

		printf("i: %d\n", i);

		SCIPrepareSegment(encodedDataSegmentsLocal[i], localAdapterNo, SCI_FLAG_DMA_SOURCE_ONLY, &error);
		sisci_assert(error);

		buffer = SCIMapLocalSegment(encodedDataSegmentsLocal[i], &encodedDataMapsLocal[i], 0, segmentSize, NULL, SCI_NO_FLAGS, &error);
		sisci_assert(error);

		keyframe[i] = (int*) ((uint8_t*)buffer + keyframe_offset);

		mb_Y[i] = (struct macroblock*) ((uint8_t*) buffer + mbOffsetY);
		mb_U[i] = (struct macroblock*) ((uint8_t*) buffer + mbOffsetU);
		mb_V[i] = (struct macroblock*) ((uint8_t*) buffer + mbOffsetV);
	}
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

static void cleanup_remote_segment(sci_remote_segment_t* segment, sci_map_t *map)
{
	sci_error_t error;

	SCIUnmapSegment(*map, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCIDisconnectSegment(*segment, SCI_NO_FLAGS, &error);
	sisci_check(error);
}

void cleanup_segments()
{
	int i;
	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
		//cleanup_local_segment(&encodedDataSegmentsLocal[i], &encodedDataMapsLocal[i]);
		cudaFree((void*)cudaBuffers[i]);
		cleanup_local_segment(&imageSegments[i], &imageMaps[i]);
		cleanup_remote_segment(&encodedDataSegmentsWriter[i], &encodedDataMapsRemote[i]);
	}

}

void receive_width_and_height(uint32_t* width, uint32_t* height)
{
	sci_error_t error;

	printf("Waiting for width and height from reader... ");
	fflush(stdout);

	uint32_t widthAndHeight[2];
	unsigned int length = 2 * sizeof(uint32_t);
	SCIWaitForDataInterrupt(interruptsFromReader[0], &widthAndHeight, &length, SCI_INFINITE_TIMEOUT,
			SCI_NO_FLAGS, &error);
	sisci_assert(error);

	*width = widthAndHeight[0];
	*height = widthAndHeight[1];
	printf("Done!\n");
}

void send_width_and_height(uint32_t width, uint32_t height) {
	sci_error_t error;

	uint32_t widthAndHeight[2] = {width, height};
	SCITriggerDataInterrupt(interruptsToWriter[0], (void*) &widthAndHeight, 2*sizeof(uint32_t), SCI_NO_FLAGS, &error);
	sisci_assert(error);
}

int wait_for_reader(int segNum)
{
	sci_error_t error;

	static unsigned int done_size = sizeof(uint8_t);
	uint8_t done;

	SCIWaitForDataInterrupt(interruptsFromReader[segNum], &done, &done_size, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	return done;
}

void wait_for_writer(int segNum)
{
	sci_error_t error;

	do {
		SCIWaitForInterrupt(interruptsFromWriter[segNum], SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	} while (error != SCI_ERR_OK);
}


sci_callback_action_t dma_callback(void *arg, sci_dma_queue_t dma_queue, sci_error_t status) {
	sci_callback_action_t retVal;

	if (status == SCI_ERR_OK) {
		// Send interrupt to computation node signaling that the frame has been transferred
		signal_writer(DATA_TRANSFERRED, *(int*)arg);

		retVal = SCI_CALLBACK_CONTINUE;
	}

	else {
		retVal = SCI_CALLBACK_CANCEL;
	}

	free(arg);

	return retVal;

}

void transfer_encoded_data(int keyframe_val, struct macroblock** mbs, dct_t* residuals, int segNum)
{
	//sci_error_t error;
	*keyframe[segNum] = keyframe_val;
	//memcpy(mb_Y[segNum], mbs[Y_COMPONENT], mbSizeY+mbSizeU+mbSizeV);

	/*
	dis_dma_vec_t dis_dma_vec[1];
	dis_dma_vec[0].size = sizeof(int)+mbSizeY+mbSizeU+mbSizeV;
	dis_dma_vec[0].local_offset = 0;
	dis_dma_vec[0].remote_offset = 0;
	dis_dma_vec[1].size = residualsSizeY+residualsSizeU+residualsSizeV;
	dis_dma_vec[1].local_offset = residualsY_offset;
	dis_dma_vec[1].remote_offset = residualsY_offset;
	//cudaMemcpy(remote_residuals_Y[segNum], residuals->Ydct, residualsSizeY+residualsSizeU+residualsSizeV, cudaMemcpyDeviceToHost);
	*/
	printf("Sending\n");
	//cudaMemcpy((void*)remote_mb_Y[segNum], mbs[Y_COMPONENT], mbSizeY+mbSizeU+mbSizeV, cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)remote_mb_Y[segNum], mbs[Y_COMPONENT], mbSizeY, cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)remote_mb_U[segNum], mbs[U_COMPONENT], mbSizeU, cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)remote_mb_V[segNum], mbs[V_COMPONENT], mbSizeV, cudaMemcpyDeviceToHost);

	//cudaMemcpy((void*)remote_residuals_Y[segNum], residuals->Ydct, residualsSizeY+residualsSizeU+residualsSizeV, cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)remote_residuals_Y[segNum], residuals->Ydct, residualsSizeY, cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)remote_residuals_U[segNum], residuals->Udct, residualsSizeU, cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)remote_residuals_V[segNum], residuals->Vdct, residualsSizeV, cudaMemcpyDeviceToHost);

	printf("sent\n");

	//int *arg = (int*) malloc(sizeof(int));
	//*arg = segNum;
	signal_writer(DATA_TRANSFERRED, segNum);
	//SCIStartDmaTransferVec(dmaQueues[segNum], encodedDataSegmentsLocal[segNum], encodedDataSegmentsWriter[segNum],
			//1, dis_dma_vec,	dma_callback, arg, SCI_FLAG_USE_CALLBACK, &error);
	//SCIStartDmaTransfer(dmaQueues[segNum], encodedDataSegmentsLocal[segNum], encodedDataSegmentsWriter[segNum],
		//	0, sizeof(int)+mbSizeY+mbSizeU+mbSizeV, 0, dma_callback, arg, SCI_FLAG_USE_CALLBACK, &error);
	//sisci_assert(error);
}

void wait_for_image_transfer(int segNum) {
	sci_error_t error;

	SCIWaitForDMAQueue(dmaQueues[segNum], SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	//sisci_assert(error);
}

void signal_reader(int segNum)
{
	sci_error_t error;
	SCITriggerInterrupt(interruptsToReader[segNum], SCI_NO_FLAGS, &error);
	sisci_assert(error);
}

void signal_writer(writer_signal signal, int segNum)
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

	SCITriggerDataInterrupt(interruptsToWriter[segNum], (void*) &data, sizeof(uint8_t), SCI_NO_FLAGS, &error);
	sisci_assert(error);
}
