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

// Interrupts
typedef enum {
	READY_FOR_ORIG_TRANSFER = 1000,
	MORE_DATA_TRANSFERRED = 15,
	ENCODED_FRAME_TRANSFERRED = 20,
	DATA_WRITTEN = 25
} c63_interrupt;

// Segments
typedef enum {
	SEGMENT_ENCODER_IMAGE = 177
} c63_segment_encoder;

typedef enum {
	SEGMENT_WRITER_ENCODED = 193
} c63_segment_writer;

struct segment_yuv
{
	const volatile uint8_t* Y;
	const volatile uint8_t* U;
	const volatile uint8_t* V;
};

#endif /* SISCI_COMMON_H_ */
