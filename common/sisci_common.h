#ifndef SISCI_COMMON_H_
#define SISCI_COMMON_H_

#include <inttypes.h>

#define SCI_NO_FLAGS        0
#define SCI_NO_CALLBACK     NULL

#define NUM_IMAGE_SEGMENTS 2

// Signals
typedef enum {
	IMAGE_TRANSFERRED,
	NO_MORE_FRAMES
} encoder_signal;

typedef enum {
	ENCODING_FINISHED,
	DATA_TRANSFERRED
} writer_signal;

// Interrupts
typedef enum {
	READY_FOR_ORIG_TRANSFER,
	MORE_DATA_TRANSFERRED,
	ENCODED_FRAME_TRANSFERRED,
	DATA_WRITTEN
} c63_interrupt;

// Segments
typedef enum {
	SEGMENT_READER_IMAGE,
	SEGMENT_READER_IMAGE2,
	SEGMENT_ENCODER_IMAGE,
	SEGMENT_ENCODER_IMAGE2,
	SEGMENT_ENCODER_ENCODED,
	SEGMENT_ENCODER_ENCODED2,
	SEGMENT_WRITER_ENCODED,
	SEGMENT_WRITER_ENCODED2
} c63_segment;

struct segment_yuv
{
	const volatile uint8_t* Y;
	const volatile uint8_t* U;
	const volatile uint8_t* V;
};

static inline uint32_t getLocalSegId(unsigned int localNodeId, unsigned int remoteNodeId, c63_segment segment)
{
	return (localNodeId << 24) | (remoteNodeId << 16) | segment;
}

static inline uint32_t getRemoteSegId(unsigned int localNodeId, unsigned int remoteNodeId, c63_segment segment)
{
	return getLocalSegId(remoteNodeId, localNodeId, segment);
}

#endif /* SISCI_COMMON_H_ */
