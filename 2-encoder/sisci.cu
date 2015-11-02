#include <sisci_api.h>
#include <sisci_error.h>

#include "../common/sisci_common.h"
#include "../common/sisci_errchk.h"
#include "sisci.h"

// Local
static unsigned int localAdapterNo;
static unsigned int localNodeId;
static sci_desc_t sd;

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
static sci_remote_segment_t encodedDataSegment;
static sci_map_t encodedDataMap;
static sci_sequence_t writerSequence;

static unsigned int keyframeSize;
static unsigned int mbSizeY;
static unsigned int mbSizeU;
static unsigned int mbSizeV;
static unsigned int residualsSizeY;
static unsigned int residualsSizeU;
static unsigned int residualsSizeV;


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

	SCIGetLocalNodeId(localAdapterNo, &localNodeId, SCI_NO_FLAGS, &error);
	sisci_assert(error);

	// Interrupts from the reader
	unsigned int interruptFromReaderNo = MORE_DATA_TRANSFERRED;
	SCICreateDataInterrupt(sd, &interruptFromReader, localAdapterNo, &interruptFromReaderNo, NULL,
			NULL, SCI_FLAG_FIXED_INTNO, &error);
	sisci_assert(error);

	// Interrupts from the writer
	unsigned int interruptFromWriterNo = DATA_WRITTEN;
	SCICreateInterrupt(sd, &interruptFromWriter, localAdapterNo, &interruptFromWriterNo, NULL,
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
		SCIConnectDataInterrupt(sd, &interruptToWriter, writerNodeId, localAdapterNo,
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

	SCIClose(sd, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCITerminate();
}

yuv_t init_image_segment(struct c63_common* cm)
{
	yuv_t image;
	unsigned int localSegmentId = (localNodeId << 16) | (readerNodeId << 8) | 0; // TODO enum elns

	unsigned int segmentSizeY = cm->ypw * cm->yph * sizeof(uint8_t);
	unsigned int segmentSizeU = cm->upw * cm->uph * sizeof(uint8_t);
	unsigned int segmentSizeV = cm->vpw * cm->vph * sizeof(uint8_t);
	unsigned int segmentSize = segmentSizeY + segmentSizeU + segmentSizeV;

	sci_error_t error;

	SCICreateSegment(sd, &imageSegment, localSegmentId, segmentSize, SCI_NO_CALLBACK, NULL, SCI_NO_FLAGS, &error);
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

void init_encoded_data_segment(struct c63_common* cm)
{
	unsigned int remoteSegmentId = (writerNodeId << 16) | (localNodeId) | 0;

	sci_error_t error;

	// Connect to remote segment on writer
	do {
		SCIConnectSegment(sd, &encodedDataSegment, writerNodeId, remoteSegmentId, localAdapterNo,
				SCI_NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, SCI_NO_FLAGS, &error);
	} while (error != SCI_ERR_OK);

	// Get segment size
	uint32_t segmentSize = SCIGetRemoteSegmentSize(encodedDataSegment);

	unsigned int offset = 0;
	SCIMapRemoteSegment(encodedDataSegment, &encodedDataMap, offset, segmentSize, NULL, SCI_NO_FLAGS, &error);
	sisci_assert(error);

    SCICreateMapSequence(encodedDataMap, &writerSequence, 0, &error);
    sisci_assert(error);

    keyframeSize = sizeof(int);
    mbSizeY = cm->mb_rowsY * cm->mb_colsY * sizeof(struct macroblock);
    mbSizeU = cm->mb_rowsUV * cm->mb_colsUV * sizeof(struct macroblock);
    mbSizeV = cm->mb_rowsUV * cm->mb_colsUV * sizeof(struct macroblock);
    residualsSizeY = cm->ypw * cm->yph * sizeof(int16_t);
    residualsSizeU = cm->upw * cm->uph * sizeof(int16_t);
    residualsSizeV = cm->vpw * cm->vph * sizeof(int16_t);
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

static void cleanup_remote_segment(sci_remote_segment_t* segment, sci_map_t* map, sci_sequence_t* sequence)
{
	sci_error_t error;

	SCIRemoveSequence(*sequence, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCIUnmapSegment(*map, SCI_NO_FLAGS, &error);
	sisci_check(error);

	SCIDisconnectSegment(*segment, SCI_NO_FLAGS, &error);
	sisci_check(error);
}

void cleanup_segments()
{
	cleanup_local_segment(&imageSegment, &imageMap);
	cleanup_remote_segment(&encodedDataSegment, &encodedDataMap, &writerSequence);
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

void transfer_encoded_data(int keyframe, struct macroblock** mbs, dct_t* residuals)
{
	sci_error_t error;
	unsigned int offset = 0;

	// TODO: These currently fail when using SCI_FLAG_ERROR_CHECK
	SCIMemCpy(writerSequence, (void*) &keyframe, encodedDataMap, offset, keyframeSize, SCI_NO_FLAGS, &error);
	sisci_assert(error);
	offset += keyframeSize;

	SCIMemCpy(writerSequence, mbs[Y_COMPONENT], encodedDataMap, offset, mbSizeY, SCI_NO_FLAGS, &error);
	sisci_assert(error);
	offset += mbSizeY;

	SCIMemCpy(writerSequence, residuals->Ydct, encodedDataMap, offset, residualsSizeY, SCI_NO_FLAGS, &error);
	sisci_assert(error);
	offset += residualsSizeY;

	SCIMemCpy(writerSequence, mbs[U_COMPONENT], encodedDataMap, offset, mbSizeU, SCI_NO_FLAGS, &error);
	sisci_assert(error);
	offset += mbSizeU;

	SCIMemCpy(writerSequence, residuals->Udct, encodedDataMap, offset, residualsSizeU, SCI_NO_FLAGS, &error);
	sisci_assert(error);
	offset += residualsSizeU;

	SCIMemCpy(writerSequence, mbs[V_COMPONENT], encodedDataMap, offset, mbSizeV, SCI_NO_FLAGS, &error);
	sisci_assert(error);
	offset += mbSizeV;

	SCIMemCpy(writerSequence, residuals->Vdct, encodedDataMap, offset, residualsSizeV, SCI_NO_FLAGS, &error);
	sisci_assert(error);
	offset += residualsSizeV;
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
