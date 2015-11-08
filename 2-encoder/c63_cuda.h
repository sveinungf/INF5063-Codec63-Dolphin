#ifndef C63_CUDA_H_
#define C63_CUDA_H_

struct c63_cuda {
	cudaStream_t streamY;
	cudaStream_t streamU;
	cudaStream_t streamV;
};

struct c63_common_gpu {
	unsigned int* sad_index_resultsY;
	unsigned int* sad_index_resultsU;
	unsigned int* sad_index_resultsV;
};

#endif /* C63_CUDA_H_ */
