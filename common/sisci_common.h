#ifndef SISCI_COMMON_H_
#define SISCI_COMMON_H_

#include <inttypes.h>

#define SCI_NO_FLAGS        0
#define SCI_NO_CALLBACK     NULL

// Signals
typedef enum {
	IMAGE_TRANSFERRED,
	NO_MORE_FRAMES
} encoder_signal;

// Interrupts
typedef enum {
	READY_FOR_ORIG_TRANSFER = 10,
	MORE_DATA_TRANSFERRED = 15,
	ENCODED_FRAME_TRANSFERRED = 20,
	DATA_WRITTEN = 25
} c63_interrupt;

// Segments
typedef enum {
	SEGMENT_ENCODER_IMAGE
} c63_segment_encoder;

typedef enum {
	SEGMENT_WRITER_ENCODED
} c63_segment_writer;

struct segment_yuv
{
	volatile uint8_t* Y;
	volatile uint8_t* U;
	volatile uint8_t* V;
};

#endif /* SISCI_COMMON_H_ */
