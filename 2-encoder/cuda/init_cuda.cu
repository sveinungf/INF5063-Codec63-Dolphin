#include "init_cuda.h"

struct c63_cuda init_c63_cuda()
{
	struct c63_cuda result;

	for (int i = 0; i < COLOR_COMPONENTS; ++i)
	{
		cudaStreamCreate(&result.stream[i]);
	}

	return result;
}

void cleanup_c63_cuda(struct c63_cuda& c63_cuda)
{
	for (int i = 0; i < COLOR_COMPONENTS; ++i)
	{
		cudaStreamDestroy(c63_cuda.stream[i]);
	}
}

static struct boundaries init_me_boundaries_gpu(const struct boundaries& indata, int cols, int rows,
		cudaStream_t stream)
{
	struct boundaries result;

	cudaMalloc((void**) &result.left, cols * sizeof(int));
	cudaMalloc((void**) &result.right, cols * sizeof(int));
	cudaMalloc((void**) &result.top, rows * sizeof(int));
	cudaMalloc((void**) &result.bottom, rows * sizeof(int));

	cudaMemcpyAsync((void*) result.left, indata.left, cols * sizeof(int), cudaMemcpyHostToDevice,
			stream);
	cudaMemcpyAsync((void*) result.right, indata.right, cols * sizeof(int), cudaMemcpyHostToDevice,
			stream);
	cudaMemcpyAsync((void*) result.top, indata.top, rows * sizeof(int), cudaMemcpyHostToDevice,
			stream);
	cudaMemcpyAsync((void*) result.bottom, indata.bottom, rows * sizeof(int),
			cudaMemcpyHostToDevice, stream);

	return result;
}

static void cleanup_me_boundaries_gpu(struct boundaries& boundaries_gpu)
{
	cudaFree((void*) boundaries_gpu.left);
	cudaFree((void*) boundaries_gpu.right);
	cudaFree((void*) boundaries_gpu.top);
	cudaFree((void*) boundaries_gpu.bottom);
}

struct c63_common_gpu init_c63_gpu(const struct c63_common* cm, const struct c63_cuda& c63_cuda)
{
	struct c63_common_gpu result;

	for (int i = 0; i < COLOR_COMPONENTS; ++i)
	{
		int cols = cm->mb_cols[i];
		int rows = cm->mb_rows[i];
		const struct boundaries& boundaries = cm->me_boundaries[i];
		cudaStream_t stream = c63_cuda.stream[i];

		result.me_boundaries[i] = init_me_boundaries_gpu(boundaries, cols, rows, stream);
		cudaMalloc(&result.sad_index_results[i], cols * rows * sizeof(unsigned int));
	}

	return result;
}

void cleanup_c63_gpu(struct c63_common_gpu& cm_gpu)
{
	for (int i = 0; i < COLOR_COMPONENTS; ++i)
	{
		cleanup_me_boundaries_gpu(cm_gpu.me_boundaries[i]);
		cudaFree(cm_gpu.sad_index_results[i]);
	}
}
