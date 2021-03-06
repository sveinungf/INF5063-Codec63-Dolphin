#ifndef SISCI_H_
#define SISCI_H_

#include "../common/sisci_common.h"


void init_SISCI(unsigned int localAdapter, unsigned int encoderNode);
void cleanup_SISCI();

void receive_width_and_height(uint32_t *width, uint32_t *height);

uint8_t *init_local_segment(uint32_t localSegmentSize, int segNum);

void signal_encoder(int segNum);
void wait_for_encoder(uint8_t *done, unsigned int *length, int segNum);

#endif /* SISCI_H_ */
