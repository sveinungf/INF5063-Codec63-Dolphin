#ifndef SISCI_H_
#define SISCI_H_

#include "../common/sisci_common.h"
#include "c63.h"


typedef enum {
	ENCODING_FINISHED,
	DATA_TRANSFERRED
} writer_signal;

void init_SISCI(unsigned int localAdapter, unsigned int readerNode, unsigned int writerNode);
void cleanup_SISCI();

void set_sizes_offsets(struct c63_common *cm);

struct segment_yuv init_image_segment(struct c63_common* cm, int segNum);
void init_remote_encoded_data_segment(int segNum);
void init_local_encoded_data_segments();
void init_local_encoded_data_segment();
void cleanup_segments();

void receive_width_and_height(uint32_t* width, uint32_t* height);
void send_width_and_height(uint32_t width, uint32_t height);

int wait_for_reader(int segNum);
void wait_for_writer(int segNum);

void transfer_encoded_data(int keyframe, struct macroblock** mbs, dct_t* residuals, int segNum);
void wait_for_image_transfer();

void signal_reader(int segNum);
void signal_writer(writer_signal signal, int segNum);

#endif /* SISCI_H_ */
