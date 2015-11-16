#include "init_cuda.h"


static const int Y = Y_COMPONENT;
static const int U = U_COMPONENT;
static const int V = V_COMPONENT;

static yuv_t* create_image_gpu(struct c63_common *cm)
{
	yuv_t* image = (yuv_t*) malloc(sizeof(yuv_t));
	cudaMalloc((void**) &image->Y, cm->ypw * cm->yph * sizeof(uint8_t));
	cudaMalloc((void**) &image->U, cm->upw * cm->uph * sizeof(uint8_t));
	cudaMalloc((void**) &image->V, cm->vpw * cm->vph * sizeof(uint8_t));

	return image;
}

static void destroy_image_gpu(yuv_t* image)
{
	cudaFree(image->Y);
	cudaFree(image->U);
	cudaFree(image->V);
	free(image);
}

void init_frame_gpu(struct c63_common* cm, struct frame* f)
{
	f->recons_gpu = create_image_gpu(cm);
	f->predicted_gpu = create_image_gpu(cm);

	f->residuals_gpu = (dct_t*) malloc(sizeof(dct_t));
	cudaMalloc((void**) &f->residuals_gpu->Ydct, cm->ypw * cm->yph * sizeof(int16_t));
	cudaMalloc((void**) &f->residuals_gpu->Udct, cm->upw * cm->uph * sizeof(int16_t));
	cudaMalloc((void**) &f->residuals_gpu->Vdct, cm->vpw * cm->vph * sizeof(int16_t));

	cudaMalloc((void**) &f->mbs_gpu[Y], cm->mb_rows[Y] * cm->mb_cols[Y] *
			sizeof(struct macroblock));
	cudaMalloc((void**) &f->mbs_gpu[U], cm->mb_rows[U] * cm->mb_cols[U] *
			sizeof(struct macroblock));
	cudaMalloc((void**) &f->mbs_gpu[V], cm->mb_rows[V] * cm->mb_cols[V] *
			sizeof(struct macroblock));
}

void deinit_frame_gpu(struct frame* f)
{
	destroy_image_gpu(f->recons_gpu);
	destroy_image_gpu(f->predicted_gpu);

	cudaFree(f->residuals_gpu->Ydct);
	cudaFree(f->residuals_gpu->Udct);
	cudaFree(f->residuals_gpu->Vdct);
	free(f->residuals_gpu);

	cudaFree(f->mbs_gpu[Y_COMPONENT]);
	cudaFree(f->mbs_gpu[U_COMPONENT]);
	cudaFree(f->mbs_gpu[V_COMPONENT]);
}

struct c63_cuda init_c63_cuda()
{
	struct c63_cuda result;

	for (int i = 0; i < COLOR_COMPONENTS; ++i)
	{
		cudaStreamCreate(&result.stream[i]);
		cudaStreamCreate(&result.memcpy_stream[i]);

		cudaEventCreate(&result.me_done[i]);
		cudaEventCreate(&result.dctquant_done[i]);
	}

	return result;
}

void cleanup_c63_cuda(struct c63_cuda& c63_cuda)
{
	for (int i = 0; i < COLOR_COMPONENTS; ++i)
	{
		cudaStreamDestroy(c63_cuda.stream[i]);
		cudaStreamDestroy(c63_cuda.memcpy_stream[i]);

		cudaEventDestroy(c63_cuda.me_done[i]);
		cudaEventDestroy(c63_cuda.dctquant_done[i]);
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
