#ifndef SISCI_H_
#define SISCI_H_

#include <sisci_api.h>
#include <sisci_error.h>

#include "sisci_common.h"



void init_SISCI(unsigned int localAdapterNo, unsigned int encoderNodeId);
void cleanup_SISCI();

void init_msg_segments();
struct segment_yuv init_image_segment(unsigned int sizeY, unsigned int sizeU, unsigned int sizeV, int segNum);
void cleanup_segments();

void send_width_and_height(uint32_t width, uint32_t height);

void wait_for_encoder(int32_t frameNum, int offset);

void transfer_image_async(int segNum, int32_t frameNum);
void wait_for_image_transfer(int segNum);

void signal_encoder(encoder_signal signal, int frameNum);

#endif /* SISCI_H_ */
