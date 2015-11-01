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

#endif /* SISCI_COMMON_H_ */
