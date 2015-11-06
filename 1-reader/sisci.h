#ifndef SISCI_H_
#define SISCI_H_

#include <sisci_api.h>
#include <sisci_error.h>

#include "../common/sisci_common.h"

struct local_segment_reader {
	sci_desc_t sd;
	sci_local_segment_t segment;
	sci_map_t map;

	struct segment_yuv images[2];

	unsigned int segmentSize;
};

struct remote_segment_reader {
	sci_desc_t *sd;
	sci_remote_segment_t segment;

	unsigned int segmentSize;
};

void init_SISCI(unsigned int localAdapterNo, unsigned int encoderNodeId);
void cleanup_SISCI();

void init_image_segments(struct segment_yuv *images, unsigned int sizeY, unsigned int sizeU, unsigned int sizeV);
void cleanup_segments();

void send_width_and_height(uint32_t width, uint32_t height);

void wait_for_encoder();

void transfer_image_async(int imgNum);
void wait_for_image_transfer();

void signal_encoder(encoder_signal signal);

#endif /* SISCI_H_ */
