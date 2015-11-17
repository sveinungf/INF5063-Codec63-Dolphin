#ifndef SISCI_H_
#define SISCI_H_

#include <sisci_api.h>
#include <sisci_error.h>

#include "sisci_common.h"



void init_SISCI(unsigned int localAdapterNo, unsigned int encoderNodeId);
void cleanup_SISCI();

struct segment_yuv init_image_segment(unsigned int sizeY, unsigned int sizeU, unsigned int sizeV, int segNum);
void cleanup_segments();

void send_width_and_height(uint32_t width, uint32_t height);

void signal_encoder(encoder_signal signal, int segNum);
void wait_for_encoder(int segNum);

void wait_for_image_transfer(int segNum);
void transfer_image_async(int imgNum);

#endif /* SISCI_H_ */
