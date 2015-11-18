#include "init_cuda.h"


static yuv_t* create_image_gpu(struct c63_common *cm)
{
	yuv_t* image = (yuv_t*) malloc(sizeof(yuv_t));
	uint8_t** components[COLOR_COMPONENTS] = { &image->Y, &image->U, &image->V };

	for (int c = 0; c < COLOR_COMPONENTS; ++c) {
		if (ON_GPU(c)) {
			size_t size = cm->padw[c] * cm->padh[c] * sizeof(uint8_t);
			cudaMalloc((void**) components[c], size);
		}
	}

	return image;
}

static void destroy_image_gpu(yuv_t* image)
{
	uint8_t* components[COLOR_COMPONENTS] = { image->Y, image->U, image->V };

	for (int c = 0; c < COLOR_COMPONENTS; ++c) {
		if (ON_GPU(c)) {
			cudaFree(components[c]);
		}
	}

	free(image);
}

void init_frame_gpu(struct c63_common* cm, struct frame* f)
{
	f->recons_gpu = create_image_gpu(cm);
	f->predicted_gpu = create_image_gpu(cm);

	f->residuals_gpu = (dct_t*) malloc(sizeof(dct_t));

	dct_t* dct = f->residuals_gpu;
	int16_t** residuals[COLOR_COMPONENTS] = { &dct->Ydct, &dct->Udct, &dct->Vdct };

	for (int c = 0; c < COLOR_COMPONENTS; ++c) {
		if (ON_GPU(c)) {
			size_t res_size = cm->padw[c] * cm->padh[c] * sizeof(int16_t);
			size_t mb_size = cm->mb_rows[c] * cm->mb_cols[c] * sizeof(struct macroblock);

			cudaMalloc((void**) residuals[c], res_size);
			cudaMalloc((void**) &f->mbs_gpu[c], mb_size);
		}
	}
}

void deinit_frame_gpu(struct frame* f)
{
	destroy_image_gpu(f->recons_gpu);
	destroy_image_gpu(f->predicted_gpu);

	dct_t* dct = f->residuals_gpu;
	int16_t* residuals[COLOR_COMPONENTS] = { dct->Ydct, dct->Udct, dct->Vdct };

	for (int c = 0; c < COLOR_COMPONENTS; ++c) {
		if (ON_GPU(c)) {
			cudaFree(residuals[c]);
			cudaFree(f->mbs_gpu[c]);
		}
	}

	free(f->residuals_gpu);
}

struct c63_cuda init_c63_cuda()
{
	struct c63_cuda result;

	for (int c = 0; c < COLOR_COMPONENTS; ++c)
	{
		cudaStreamCreate(&result.memcpy_stream[c]);

		if (ON_GPU(c)) {
			cudaStreamCreate(&result.stream[c]);

			cudaEventCreate(&result.me_done[c]);
			cudaEventCreate(&result.dctquant_done[c]);
		}
	}

	return result;
}

void cleanup_c63_cuda(struct c63_cuda& c63_cuda)
{
	for (int c = 0; c < COLOR_COMPONENTS; ++c)
	{
		cudaStreamDestroy(c63_cuda.memcpy_stream[c]);

		if (ON_GPU(c)) {
			cudaStreamDestroy(c63_cuda.stream[c]);

			cudaEventDestroy(c63_cuda.me_done[c]);
			cudaEventDestroy(c63_cuda.dctquant_done[c]);
		}
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

	for (int c = 0; c < COLOR_COMPONENTS; ++c)
	{
		if (ON_GPU(c)) {
			int cols = cm->mb_cols[c];
			int rows = cm->mb_rows[c];
			const struct boundaries& boundaries = cm->me_boundaries[c];
			cudaStream_t stream = c63_cuda.stream[c];

			result.me_boundaries[c] = init_me_boundaries_gpu(boundaries, cols, rows, stream);
			cudaMalloc(&result.sad_index_results[c], cols * rows * sizeof(unsigned int));
		}
	}

	return result;
}

void cleanup_c63_gpu(struct c63_common_gpu& cm_gpu)
{
	for (int c = 0; c < COLOR_COMPONENTS; ++c)
	{
		if (ON_GPU(c)) {
			cleanup_me_boundaries_gpu(cm_gpu.me_boundaries[c]);
			cudaFree(cm_gpu.sad_index_results[c]);
		}
	}
}
