#include <sisci_api.h>
#include <sisci_error.h>

#include "sisci.h"
#include "sisci_errchk.h"


#define MIN_SEG_SZ 237569

// Local
static unsigned int localAdapterNo;
static unsigned int localNodeId;
static sci_desc_t reader_sds[NUM_IMAGE_SEGMENTS];
static sci_desc_t writer_sds[NUM_IMAGE_SEGMENTS];
static volatile uint8_t* cudaBuffers[NUM_IMAGE_SEGMENTS];

// Reader
static unsigned int readerNodeId;
static sci_local_data_interrupt_t interruptsFromReader[NUM_IMAGE_SEGMENTS];
static sci_remote_interrupt_t interruptsToReader[NUM_IMAGE_SEGMENTS];
static sci_local_segment_t imageSegments[NUM_IMAGE_SEGMENTS];
static sci_map_t imageMaps[NUM_IMAGE_SEGMENTS];

// Writer
static unsigned int segmentSizeWriter;
static unsigned int writerNodeId;

static sci_remote_data_interrupt_t interruptsToWriter[NUM_IMAGE_SEGMENTS];
static sci_local_interrupt_t interruptsFromWriter[NUM_IMAGE_SEGMENTS];

static sci_local_segment_t encodedDataSegmentsLocal[NUM_IMAGE_SEGMENTS];
static sci_remote_segment_t encodedDataSegmentsWriter[NUM_IMAGE_SEGMENTS];
static sci_map_t encodedDataMapsLocal[NUM_IMAGE_SEGMENTS];

static sci_dma_queue_t dmaQueues[NUM_IMAGE_SEGMENTS];
static int callback_arg[NUM_IMAGE_SEGMENTS];

static volatile int transfer_completed[NUM_IMAGE_SEGMENTS] = {1, 1};

static unsigned int keyframeSize;
static unsigned int mbSizes[COLOR_COMPONENTS];
static unsigned int residualsSizes[COLOR_COMPONENTS];

static unsigned int keyframe_offset;
static unsigned int mbOffsets[COLOR_COMPONENTS];
static unsigned int residualsOffsets[COLOR_COMPONENTS];

static int *keyframe[NUM_IMAGE_SEGMENTS];
static struct macroblock *mb_Y[NUM_IMAGE_SEGMENTS];
static struct macroblock *mb_U[NUM_IMAGE_SEGMENTS];
static struct macroblock *mb_V[NUM_IMAGE_SEGMENTS];
static dct_t *residuals_Y[NUM_IMAGE_SEGMENTS];
static dct_t *residuals_U[NUM_IMAGE_SEGMENTS];
static dct_t *residuals_V[NUM_IMAGE_SEGMENTS];


/*
 * Initializes SISCI handles
 */
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
	printf("Connecting to interrupts on writer... ");
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

/*
 * Removes interrupts and descriptors
 */
void cleanup_SISCI()
{
	sci_error_t error;

	int i;
	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
		SCIDisconnectInterrupt(interruptsToReader[i], SCI_NO_FLAGS, &error);
		sisci_check(error);

		do {
			SCIRemoveInterrupt(interruptsFromWriter[i], SCI_NO_FLAGS, &error);
		} while (error != SCI_ERR_OK);

		SCIDisconnectDataInterrupt(interruptsToWriter[i], SCI_NO_FLAGS, &error);
		sisci_check(error);

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


/*
 * Set various sizes and offsets
 */
void set_sizes_offsets(struct c63_common *cm) {
	static const int Y = Y_COMPONENT;
	static const int U = U_COMPONENT;
	static const int V = V_COMPONENT;

    keyframeSize = sizeof(int);
    mbSizes[Y] = cm->mb_rows[Y] * cm->mb_cols[Y] * sizeof(struct macroblock);
    mbSizes[U] = cm->mb_rows[U] * cm->mb_cols[U] * sizeof(struct macroblock);
    mbSizes[V] = cm->mb_rows[V] * cm->mb_cols[V] * sizeof(struct macroblock);
    residualsSizes[Y] = cm->ypw * cm->yph * sizeof(int16_t);
    residualsSizes[U] = cm->upw * cm->uph * sizeof(int16_t);
    residualsSizes[V] = cm->vpw * cm->vph * sizeof(int16_t);

    keyframe_offset = 0;
    mbOffsets[Y] = keyframe_offset + keyframeSize;
    mbOffsets[U] = mbOffsets[Y] + mbSizes[Y];
    mbOffsets[V] = mbOffsets[U] + mbSizes[U];
    residualsOffsets[Y] = mbOffsets[V] + mbSizes[V];
    residualsOffsets[U] = residualsOffsets[Y] + residualsSizes[Y];
    residualsOffsets[V] = residualsOffsets[U] + residualsSizes[U];

}

/*
 * Initializes an image segment used for transfers between the reader and the encoder
 */
struct segment_yuv init_image_segment(struct c63_common* cm, int segNum)
{
	struct segment_yuv image;
	uint32_t localSegmentId = getLocalSegId(localNodeId, readerNodeId, (c63_segment)(SEGMENT_ENCODER_IMAGE + segNum));

	unsigned int imageSizeY = cm->ypw * cm->yph * sizeof(uint8_t);
	unsigned int imageSizeU = cm->upw * cm->uph * sizeof(uint8_t);
	unsigned int imageSizeV = cm->vpw * cm->vph * sizeof(uint8_t);
	unsigned int imageSize = imageSizeY + imageSizeU + imageSizeV;
	unsigned int segmentSize = imageSize;

	/*
	 * NOTE: Could not encode foreman.yuv without performing this check
	 */
	if(segmentSize < MIN_SEG_SZ) {
		segmentSize = MIN_SEG_SZ;
	}

	sci_error_t error;
	SCICreateSegment(reader_sds[segNum], &imageSegments[segNum], localSegmentId, segmentSize, SCI_NO_CALLBACK, NULL, SCI_FLAG_EMPTY, &error);
	sisci_assert(error);

	/*
	 * NOTE: Could not encode foreman.yuv without tripling the CUDA segment size
	 * The SISCI API reported error 904 - "out of local resources"
	 * The large video files worked without problems
	 */
	cudaMalloc((void**)&cudaBuffers[segNum], 3*segmentSize);

	SCIAttachPhysicalMemory(0, (void*)cudaBuffers[segNum], 0, segmentSize, imageSegments[segNum], SCI_FLAG_CUDA_BUFFER, &error);
	sisci_assert(error);

	void* buffer = SCIMapLocalSegment(imageSegments[segNum], &imageMaps[segNum], 0, segmentSize, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	unsigned int offset = 0;
	image.Y = (uint8_t*) buffer + offset;
	offset += imageSizeY;
	image.U = (uint8_t*) buffer + offset;
	offset += imageSizeU;
	image.V = (uint8_t*) buffer + offset;
	offset += imageSizeV;

	SCIPrepareSegment(imageSegments[segNum], localAdapterNo, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCISetSegmentAvailable(imageSegments[segNum], localAdapterNo, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	return image;
}

/*
 * Connects to a remote segment on the writer
 */
void init_remote_encoded_data_segment(int segNum)
{
	uint32_t remoteSegmentId = getRemoteSegId(localNodeId, writerNodeId, (c63_segment)(SEGMENT_WRITER_ENCODED + segNum));

	sci_error_t error;

	// Connect to remote segment on writer
	do {
		SCIConnectSegment(writer_sds[segNum], &encodedDataSegmentsWriter[segNum], writerNodeId, remoteSegmentId, localAdapterNo,
				SCI_NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	} while (error != SCI_ERR_OK);

	// Get segment size
	segmentSizeWriter = SCIGetRemoteSegmentSize(encodedDataSegmentsWriter[segNum]);
}

/*
 * Initializes the local SISCI segments on the encoder
 */
void init_local_encoded_data_segments() {
	sci_error_t error;
	uint32_t localSegmentId;
	void *buffer;

	int i;
	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
		localSegmentId = getLocalSegId(localNodeId, writerNodeId, (c63_segment)(SEGMENT_ENCODER_ENCODED + i));

		SCICreateSegment(writer_sds[i], &encodedDataSegmentsLocal[i], localSegmentId, segmentSizeWriter, SCI_NO_CALLBACK, NULL, SCI_NO_FLAGS, &error);
		sisci_assert(error);

		SCIPrepareSegment(encodedDataSegmentsLocal[i], localAdapterNo, SCI_FLAG_DMA_SOURCE_ONLY, &error);
		sisci_assert(error);

		buffer = SCIMapLocalSegment(encodedDataSegmentsLocal[i], &encodedDataMapsLocal[i], 0, segmentSizeWriter, NULL, SCI_NO_FLAGS, &error);
		sisci_assert(error);

		keyframe[i] = (int*) ((uint8_t*)buffer + keyframe_offset);

		mb_Y[i] = (struct macroblock*) ((uint8_t*) buffer + mbOffsets[Y_COMPONENT]);
		mb_U[i] = (struct macroblock*) ((uint8_t*) buffer + mbOffsets[U_COMPONENT]);
		mb_V[i] = (struct macroblock*) ((uint8_t*) buffer + mbOffsets[V_COMPONENT]);

		residuals_Y[i] = (dct_t*) ((uint8_t*) buffer + residualsOffsets[Y_COMPONENT]);
		residuals_U[i] = (dct_t*) ((uint8_t*) buffer + residualsOffsets[U_COMPONENT]);
		residuals_V[i] = (dct_t*) ((uint8_t*) buffer + residualsOffsets[V_COMPONENT]);
	}
}

/*
 * Cleans up local SISCI segments
 */
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

/*
 * Disconnects a remote segment
 */
static void cleanup_remote_segment(sci_remote_segment_t* segment)
{
	sci_error_t error;

	SCIDisconnectSegment(*segment, SCI_NO_FLAGS, &error);
	sisci_check(error);
}

/*
 * Cleans up local and remote SISCI segments
 * Frees the memory on the GPU
 */
void cleanup_segments()
{
	int i;
	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
		cleanup_local_segment(&encodedDataSegmentsLocal[i], &encodedDataMapsLocal[i]);
		cudaFree((void*)cudaBuffers[i]);
		cleanup_local_segment(&imageSegments[i], &imageMaps[i]);
		cleanup_remote_segment(&encodedDataSegmentsWriter[i]);
	}

}

/*
 * Receives the video file dimensions from the reader
 */
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

/*
 * Sends the video file dimensions to the writer
 */
void send_width_and_height(uint32_t width, uint32_t height) {
	sci_error_t error;

	uint32_t widthAndHeight[2] = {width, height};
	SCITriggerDataInterrupt(interruptsToWriter[0], (void*) &widthAndHeight, 2*sizeof(uint32_t), SCI_NO_FLAGS, &error);
	sisci_assert(error);
}

/*
 * Waits until the previous segment transfer has completed
 */
void wait_for_image_transfer(int segNum) {
	while(!transfer_completed[segNum]);
	transfer_completed[segNum] = 0;
}

/*
 * Copies data from the temporary buffers into the SISCI segment used to
 * transfer data to the writer
 */
void copy_to_segment(struct macroblock **mbs, dct_t* residuals, int segNum) {
	memcpy(mb_Y[segNum], mbs[Y_COMPONENT], mbSizes[Y_COMPONENT]);
	memcpy(mb_U[segNum], mbs[U_COMPONENT], mbSizes[U_COMPONENT]);
	memcpy(mb_V[segNum], mbs[V_COMPONENT], mbSizes[V_COMPONENT]);

	memcpy(residuals_Y[segNum], residuals->Ydct, residualsSizes[Y_COMPONENT]);
	memcpy(residuals_U[segNum], residuals->Udct, residualsSizes[U_COMPONENT]);
	memcpy(residuals_V[segNum], residuals->Vdct, residualsSizes[V_COMPONENT]);
}


/*
 * Callback function for signalling the completion of a DMA transfer
 */
static sci_callback_action_t dma_callback(void *arg, sci_dma_queue_t, sci_error_t status) {
	sci_callback_action_t retVal;

	if (status == SCI_ERR_OK) {
		// Send interrupt to computation node signalling that the frame has been transferred
		signal_writer(DATA_TRANSFERRED, *(int*)arg);

		transfer_completed[*(int*)arg] = 1;

		retVal = SCI_CALLBACK_CONTINUE;
	}

	else {
		retVal = SCI_CALLBACK_CANCEL;
	}

	return retVal;
}

/*
 * Transfers data asynchronously to the writer
 * A callback is used to signal the writer about the completion
 */
void transfer_encoded_data(int keyframe_val, int segNum)
{
	sci_error_t error;
	*keyframe[segNum] = keyframe_val;

	callback_arg[segNum] = segNum;

	SCIStartDmaTransfer(dmaQueues[segNum], encodedDataSegmentsLocal[segNum], encodedDataSegmentsWriter[segNum], 0, segmentSizeWriter, 0, dma_callback, &callback_arg[segNum], SCI_FLAG_USE_CALLBACK, &error);
	sisci_assert(error);

}


/*
 * Signals the reader that the encoder is ready for a new image
 */
void signal_reader(int segNum)
{
	sci_error_t error;

	SCITriggerInterrupt(interruptsToReader[segNum], SCI_NO_FLAGS, &error);
	sisci_assert(error);
}


/*
 * Waits until the reader signals that it has transferred a new image
 */
int wait_for_reader(int segNum)
{
	sci_error_t error;

	static unsigned int done_size = sizeof(uint8_t);
	uint8_t done;

	SCIWaitForDataInterrupt(interruptsFromReader[segNum], &done, &done_size, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	return done;
}


/*
 * Signals the writer that new data has been transferred
 * Invoked by the callback function
 */
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


/*
 * Waits until the writer signals that it is ready for more data
 */
void wait_for_writer(int segNum)
{
	sci_error_t error;

	SCIWaitForInterrupt(interruptsFromWriter[segNum], SCI_INFINITE_TIMEOUT,
			SCI_NO_FLAGS, &error);
	sisci_assert(error);
}
