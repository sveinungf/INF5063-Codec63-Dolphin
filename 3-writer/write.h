#ifndef C63_WRITE_H_
#define C63_WRITE_H_

#include <vector>

#include "c63.h"

void write_frame_to_buffer(struct c63_common *cm, std::vector<uint8_t>& byte_vector, int threadNum);

void write_buffer_to_file(const std::vector<uint8_t>& byte_vector, FILE* file);

#endif  /* C63_WRITE_H_ */
