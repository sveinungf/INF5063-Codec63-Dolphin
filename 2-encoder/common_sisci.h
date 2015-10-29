#ifndef COMMON_SISCI_H_
#define COMMON_SISCI_H_

#define NO_FLAGS        0
#define NO_CALLBACK     NULL

// Interrupts
typedef enum {
	READY_FOR_ORIG_TRANSFER = 10,
	MORE_DATA_TRANSFERED = 15
} c63_interrupt;

#endif /* COMMON_SISCI_H_ */
