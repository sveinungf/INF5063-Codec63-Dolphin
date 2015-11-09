#ifndef C63_CUDA_H_
#define C63_CUDA_H_

#include "c63.h"

struct c63_cuda {
	cudaStream_t stream[COLOR_COMPONENTS];
};

struct c63_common_gpu {
	struct boundaries me_boundaries[COLOR_COMPONENTS];
	unsigned int* sad_index_results[COLOR_COMPONENTS];
};

#endif /* C63_CUDA_H_ */
