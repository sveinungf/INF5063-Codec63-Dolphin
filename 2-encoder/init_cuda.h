#ifndef INIT_CUDA_H_
#define INIT_CUDA_H_

#include "c63.h"

struct boundaries init_me_boundaries_gpu(struct boundaries* indata, int cols, int rows, cudaStream_t stream);
void cleanup_me_boundaries_gpu(struct boundaries* boundaries_gpu);

#endif /* INIT_CUDA_H_ */
