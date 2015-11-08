#include "init_cuda.h"

struct boundaries init_me_boundaries_gpu(struct boundaries* indata, int cols, int rows, cudaStream_t stream)
{
	struct boundaries result;

	cudaMalloc((void**) &result.left, cols * sizeof(int));
	cudaMalloc((void**) &result.right, cols * sizeof(int));
	cudaMalloc((void**) &result.top, rows * sizeof(int));
	cudaMalloc((void**) &result.bottom, rows * sizeof(int));

	cudaMemcpyAsync((void*) result.left, indata->left, cols * sizeof(int), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync((void*) result.right, indata->right, cols * sizeof(int), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync((void*) result.top, indata->top, rows * sizeof(int), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync((void*) result.bottom, indata->bottom, rows * sizeof(int), cudaMemcpyHostToDevice, stream);

	return result;
}

void cleanup_me_boundaries_gpu(struct boundaries* boundaries_gpu)
{
	cudaFree((void*) boundaries_gpu->left);
	cudaFree((void*) boundaries_gpu->right);
	cudaFree((void*) boundaries_gpu->top);
	cudaFree((void*) boundaries_gpu->bottom);
}
