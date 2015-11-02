#ifndef SISCI_COMMON_H_
#define SISCI_COMMON_H_

#define SCI_NO_FLAGS        0
#define SCI_NO_CALLBACK     NULL

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

#endif /* SISCI_COMMON_H_ */
