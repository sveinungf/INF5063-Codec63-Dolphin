#include <sisci_api.h>
#include <sisci_error.h>

#include "sisci.h"
#include "sisci_errchk.h"

#include <string.h>


#define MIN_SEG_SZ 237569

// Local
static unsigned int localAdapterNo;
static unsigned int localNodeId;
sci_desc_t reader_sds[NUM_IMAGE_SEGMENTS];
sci_desc_t writer_sds[NUM_IMAGE_SEGMENTS];
uint8_t* cudaBuffers[NUM_IMAGE_SEGMENTS];

// Reader
unsigned int readerNodeId;
sci_local_segment_t imageSegments[NUM_IMAGE_SEGMENTS];
sci_map_t imageMaps[NUM_IMAGE_SEGMENTS];

static local_syn_t reader_syn;
static remote_ack_t reader_ack;
static remote_syn_t writer_syn;
static local_ack_t writer_ack;


// Writer
static unsigned int segmentSizeWriter;
static unsigned int writerNodeId;
static int callback_arg[NUM_IMAGE_SEGMENTS];
static sci_remote_segment_t encodedDataSegmentsWriter[NUM_IMAGE_SEGMENTS];
static sci_local_segment_t encodedDataSegmentsLocal[NUM_IMAGE_SEGMENTS];
static sci_map_t encodedDataMapsLocal[NUM_IMAGE_SEGMENTS];
static sci_dma_queue_t dmaQueues[NUM_IMAGE_SEGMENTS];

static int transfer_completed[NUM_IMAGE_SEGMENTS] = {1, 1};

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

	SCIOpen(&reader_syn.sd, SCI_NO_FLAGS, &error);
	sisci_assert(error);
	SCIOpen(&reader_ack.sd, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCIOpen(&writer_syn.sd, SCI_NO_FLAGS, &error);
	sisci_assert(error);
	SCIOpen(&writer_ack.sd, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCIGetLocalNodeId(localAdapterNo, &localNodeId, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	unsigned int maxEntries = 1;
	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
		SCICreateDMAQueue(writer_sds[i], &dmaQueues[i], localAdapterNo, maxEntries, SCI_NO_FLAGS, &error);
		sisci_assert(error);
	}
}

void cleanup_SISCI()
{
	sci_error_t error;

	int i;
	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {

		SCIRemoveDMAQueue(dmaQueues[i], SCI_NO_FLAGS, &error);

		SCIClose(reader_sds[i], SCI_NO_FLAGS, &error);
		sisci_check(error);

		SCIClose(writer_sds[i], SCI_NO_FLAGS, &error);
		sisci_check(error);
	}

	SCIClose(reader_syn.sd, SCI_NO_FLAGS, &error);
	sisci_check(error);
	SCIClose(reader_ack.sd, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCIClose(writer_syn.sd, SCI_NO_FLAGS, &error);
	sisci_check(error);
	SCIClose(writer_ack.sd, SCI_NO_FLAGS, &error);
	sisci_check(error);
	SCITerminate();
}


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

struct segment_yuv init_image_segment(struct c63_common* cm, int segNum)
{
	struct segment_yuv image;
	uint32_t localSegmentId = getLocalSegId(localNodeId, readerNodeId, (c63_segment)(SEGMENT_ENCODER_IMAGE + segNum));

	unsigned int imageSizeY = cm->ypw * cm->yph * sizeof(uint8_t);
	unsigned int imageSizeU = cm->upw * cm->uph * sizeof(uint8_t);
	unsigned int imageSizeV = cm->vpw * cm->vph * sizeof(uint8_t);
	unsigned int imageSize = imageSizeY + imageSizeU + imageSizeV;
	unsigned int segmentSize = imageSize;

	if(segmentSize < MIN_SEG_SZ) {
		segmentSize = MIN_SEG_SZ;
	}

	sci_error_t error;
	SCICreateSegment(reader_sds[segNum], &imageSegments[segNum], localSegmentId, segmentSize, SCI_NO_CALLBACK, NULL, SCI_FLAG_EMPTY, &error);
	sisci_assert(error);

	cudaMalloc((void**)&cudaBuffers[segNum], 3*segmentSize);

	SCIAttachPhysicalMemory(0, cudaBuffers[segNum], 0, segmentSize, imageSegments[segNum], SCI_FLAG_CUDA_BUFFER, &error);
	sisci_assert(error);

	void* buffer = SCIMapLocalSegment(imageSegments[segNum], &imageMaps[segNum], 0, segmentSize, NULL, SCI_FLAG_READONLY_MAP, &error);
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


void init_msg_segments() {
	sci_error_t error;
	unsigned int segmentSize = sizeof(message_t);


	// READER //
	// Syn segment
	uint32_t localSegmentId = getLocalSegId(localNodeId, readerNodeId, SEGMENT_SYN);
	SCICreateSegment(reader_syn.sd, &reader_syn.segment, localSegmentId, segmentSize, SCI_NO_CALLBACK, NULL,
			SCI_NO_FLAGS, &error);
	sisci_assert(error);

	reader_syn.msg = (message_t*)SCIMapLocalSegment(reader_syn.segment, &reader_syn.map, 0, segmentSize, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);
	memset((void*)reader_syn.msg, -1, sizeof(message_t));

	SCIPrepareSegment(reader_syn.segment, localAdapterNo, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCISetSegmentAvailable(reader_syn.segment, localAdapterNo, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	//// Ack segment
	uint32_t remoteSegmentId = getRemoteSegId(localNodeId, readerNodeId, SEGMENT_ACK);
	do {
		SCIConnectSegment(reader_ack.sd, &reader_ack.segment, readerNodeId, remoteSegmentId, localAdapterNo,
				SCI_NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	} while (error != SCI_ERR_OK);

	SCIMapRemoteSegment(reader_ack.segment, &reader_ack.map, 0, segmentSize, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCICreateMapSequence(reader_ack.map, &reader_ack.sequence, SCI_NO_FLAGS, &error);
	sisci_assert(error);


	// WRITER //
	// Syn segment
	remoteSegmentId = getRemoteSegId(localNodeId, writerNodeId, SEGMENT_SYN);
	do {
		SCIConnectSegment(writer_syn.sd, &writer_syn.segment, writerNodeId, remoteSegmentId, localAdapterNo,
				SCI_NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	} while (error != SCI_ERR_OK);

	SCIMapRemoteSegment(writer_syn.segment, &writer_syn.map, 0, segmentSize, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCICreateMapSequence(writer_syn.map, &writer_syn.sequence, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	// Ack segment
	localSegmentId = getLocalSegId(localNodeId, writerNodeId, SEGMENT_ACK);
	SCICreateSegment(writer_ack.sd, &writer_ack.segment, localSegmentId, segmentSize, SCI_NO_CALLBACK, NULL,
			SCI_NO_FLAGS, &error);
	sisci_assert(error);

	writer_ack.msg = (message_t*)SCIMapLocalSegment(writer_ack.segment, &writer_ack.map, 0, segmentSize, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);
	memset((void*)writer_ack.msg, -1, sizeof(message_t));

	SCIPrepareSegment(writer_ack.segment, localAdapterNo, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	SCISetSegmentAvailable(writer_ack.segment, localAdapterNo, SCI_NO_FLAGS, &error);
	sisci_assert(error);


}

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
	int i;
	for (i = 0; i < NUM_IMAGE_SEGMENTS; ++i) {
		cleanup_local_segment(&encodedDataSegmentsLocal[i], &encodedDataMapsLocal[i]);
		cudaFree((void*)cudaBuffers[i]);
		cleanup_local_segment(&imageSegments[i], &imageMaps[i]);
		cleanup_remote_segment(&encodedDataSegmentsWriter[i]);
	}

	cleanup_local_segment(&reader_syn.segment, &reader_syn.map);
	cleanup_local_segment(&writer_ack.segment, &writer_ack.map);

	sci_error_t error;
	SCIRemoveSequence(reader_ack.sequence, SCI_NO_FLAGS, &error);
	sisci_check(error);
	SCIUnmapSegment(reader_ack.map, SCI_NO_FLAGS, &error);
	sisci_check(error);
	cleanup_remote_segment(&reader_ack.segment);
	sisci_check(error);


	SCIRemoveSequence(writer_syn.sequence, SCI_NO_FLAGS, &error);
	sisci_check(error);
	SCIUnmapSegment(writer_syn.map, SCI_NO_FLAGS, &error);
	sisci_check(error);
	cleanup_remote_segment(&writer_syn.segment);
	sisci_check(error);

}

void receive_width_and_height(uint32_t* width, uint32_t* height)
{
	sci_error_t error;

	printf("Waiting for width and height from reader... ");
	fflush(stdout);


	SCIWaitForLocalSegmentEvent(reader_syn.segment, &readerNodeId, &localAdapterNo, SCI_INFINITE_TIMEOUT,
			SCI_NO_FLAGS, &error);

	*width = reader_syn.msg->frameNum;
	*height = reader_syn.msg->status;

	reader_syn.msg->frameNum = -1;
	reader_syn.msg->status = -1;

	printf("Done!\n");
}

void send_width_and_height(uint32_t width, uint32_t height) {
	sci_error_t error;
	uint32_t widthAndHeight[2] = { width, height };
	SCIMemCpy(writer_syn.sequence, &widthAndHeight, writer_syn.map, 0, sizeof(message_t), SCI_FLAG_ERROR_CHECK, &error);
	sisci_assert(error);
}


int wait_for_reader(int32_t frameNum)
{
	sci_error_t error;

	do {
		SCIWaitForLocalSegmentEvent(reader_syn.segment, &readerNodeId, &localAdapterNo, 1,
				SCI_NO_FLAGS, &error);

		if(reader_syn.msg->status == 1) {
			return 1;
		}
		else if (reader_syn.msg->frameNum >= frameNum) {
			break;
		}
	} while(error != SCI_ERR_OK);

	return 0;
}


void wait_for_writer(int32_t frameNum, int offset)
{
	sci_error_t error;
	do {
		SCIWaitForLocalSegmentEvent(writer_ack.segment, &writerNodeId, &localAdapterNo, 1,
				SCI_NO_FLAGS, &error);
		if (writer_ack.msg->frameNum >= (frameNum-offset)) {
			break;
		}
	} while(error != SCI_ERR_OK);
}


static sci_callback_action_t dma_callback(void *arg, sci_dma_queue_t dma_queue, sci_error_t status) {
	sci_callback_action_t retVal;

	if (status == SCI_ERR_OK) {
		// Send interrupt to computation node signalling that the frame has been transferred
		signal_writer(DATA_TRANSFERRED, ((int32_t*)arg)[1]);

		transfer_completed[((int*)arg)[0]] = 1;

		retVal = SCI_CALLBACK_CONTINUE;
	}

	else {
		retVal = SCI_CALLBACK_CANCEL;
	}

	free(arg);
	return retVal;

}

void copy_to_segment(int keyframe_val, struct macroblock **mbs, dct_t* residuals, int segNum) {
	*keyframe[segNum] = keyframe_val;

	memcpy(mb_Y[segNum], mbs[Y_COMPONENT], mbSizes[Y_COMPONENT]);
	memcpy(mb_U[segNum], mbs[U_COMPONENT], mbSizes[U_COMPONENT]);
	memcpy(mb_V[segNum], mbs[V_COMPONENT], mbSizes[V_COMPONENT]);

	memcpy(residuals_Y[segNum], residuals->Ydct, residualsSizes[Y_COMPONENT]);
	memcpy(residuals_U[segNum], residuals->Udct, residualsSizes[U_COMPONENT]);
	memcpy(residuals_V[segNum], residuals->Vdct, residualsSizes[V_COMPONENT]);
}

void cuda_copy_to_segment(struct c63_common *cm, int segNum) {
	cudaMemcpy(mb_Y[segNum], cm->curframe->mbs_gpu[Y_COMPONENT],
			mbSizes[Y_COMPONENT]+mbSizes[U_COMPONENT]+mbSizes[V_COMPONENT], cudaMemcpyDeviceToHost);

	cudaMemcpy(residuals_Y[segNum], cm->curframe->residuals_gpu->Ydct,
			residualsSizes[Y_COMPONENT]+residualsSizes[U_COMPONENT]+residualsSizes[V_COMPONENT], cudaMemcpyDeviceToHost);
}

void transfer_encoded_data(int segNum, int32_t frameNum)
{
	sci_error_t error;

	//callback_arg[segNum] = segNum;
	int32_t *arg = (int32_t*)malloc(2*sizeof(int32_t));
	arg[0] = segNum;
	arg[1] = frameNum;

	SCIStartDmaTransfer(dmaQueues[segNum], encodedDataSegmentsLocal[segNum], encodedDataSegmentsWriter[segNum], 0, segmentSizeWriter, 0, dma_callback, arg, SCI_FLAG_USE_CALLBACK, &error);
	sisci_assert(error);

}

void wait_for_image_transfer(int segNum) {
	while(!transfer_completed[segNum]);
	transfer_completed[segNum] = 0;
}
/*
void wait_for_image_transfer(int segNum) {
	sci_error_t error;

	SCIWaitForDMAQueue(dmaQueues[segNum], SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	sisci_assert(error);
}
*/


void signal_reader(int32_t frameNum)
{
	sci_error_t error;

	message_t msg;
	msg.frameNum = frameNum;

	SCIMemCpy(reader_ack.sequence, &msg, reader_ack.map, 0, sizeof(message_t), SCI_FLAG_ERROR_CHECK, &error);
	sisci_assert(error);
}


void signal_writer(writer_signal signal, int32_t frameNum)
{
	sci_error_t error;

	message_t msg;
	msg.frameNum = frameNum;

	switch (signal) {
		case DATA_TRANSFERRED:
			msg.status = 0;
			break;
		case ENCODING_FINISHED:
			msg.status = 1;
			break;
	}

	SCIMemCpy(writer_syn.sequence, &msg, writer_syn.map, 0, sizeof(message_t), SCI_FLAG_ERROR_CHECK, &error);
	sisci_assert(error);
}
