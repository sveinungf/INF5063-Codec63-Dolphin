#ifndef SISCI_H_
#define SISCI_H_

#include "c63.h"
#include "sisci_common.h"



void init_SISCI(unsigned int localAdapter, unsigned int readerNode, unsigned int writerNode);
void cleanup_SISCI();

void set_sizes_offsets(struct c63_common *cm);

struct segment_yuv init_image_segment(struct c63_common* cm, int segNum);
void init_msg_segments();
void init_remote_encoded_data_segment(int segNum);
void get_pointers(struct frame *frame, int segNum);
void init_local_encoded_data_segments();
void cleanup_segments();

void receive_width_and_height(uint32_t* width, uint32_t* height);
void send_width_and_height(uint32_t width, uint32_t height);

int wait_for_reader(int32_t frameNum);
void wait_for_writer(int32_t frameNum, int offset);

void copy_to_segment(int keyframe_val, struct macroblock **mbs, dct_t* residuals, int segNum);
void cuda_copy_to_segment(struct c63_common *cm, int segNum);
void transfer_encoded_data(int segNum, int32_t frameNum);
void wait_for_image_transfer(int segNum);

void signal_reader(int32_t frameNum);
void signal_writer(writer_signal signal, int32_t frameNum);

#endif /* SISCI_H_ */
