#ifndef SISCI_H_
#define SISCI_H_

#include "../common/sisci_common.h"

void init_SISCI(unsigned int localAdapterNo, unsigned int encoderNodeId);
void cleanup_SISCI();

struct segment_yuv init_image_segment(unsigned int sizeY, unsigned int sizeU, unsigned int sizeV);
void cleanup_segments();

void send_width_and_height(uint32_t width, uint32_t height);

void wait_for_encoder();

void transfer_image_async();
void wait_for_image_transfer();

void signal_encoder(encoder_signal signal);

#endif /* SISCI_H_ */
