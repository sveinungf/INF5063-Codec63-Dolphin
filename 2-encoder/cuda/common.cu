#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"

static const int on_gpu[COLOR_COMPONENTS] = { Y_ON_GPU, U_ON_GPU, V_ON_GPU };

static const int Y = Y_COMPONENT;
static const int U = U_COMPONENT;
static const int V = V_COMPONENT;

__device__
static void dct_quant_block_8x8(float* in_data, float *out_data, int16_t __restrict__ *global_out, const uint8_t* __restrict__ quant_tbl, int i, const int j)
{
	// First dct_1d - mb = mb2 - and transpose
	float dct = 0;
	int k;
	#pragma unroll
	for (k = 0; k < 8; ++k) {
		dct += in_data[j*8+k] * dct_lookup_trans[i*8+k];
	}
	out_data[i*8+j] = dct;
	__syncthreads();

	// Second dct_1d - mb = mb2 - and transpose
	dct = 0;
	#pragma unroll
	for (k = 0; k < 8; ++k) {
		dct += out_data[j*8+k] * dct_lookup_trans[i*8+k];
	}

	// Scale
	if(i == 0) {
		dct *= ISQRT2;
	}
	if(j == 0) {
		dct *= ISQRT2;
	}
	in_data[i*8+j] = dct;
	__syncthreads();

	// Quantize and set value in out_data
	dct = in_data[UV_indexes[i*8+j]];
	global_out[i*8+j] = round((dct/4.0f) / quant_tbl[i*8+j]);
}

__global__
void dct_quantize(const uint8_t* __restrict__ in_data, const uint8_t* __restrict__ prediction, int w, int16_t* __restrict__ out_data, int quantization)
{
	const int block_offset = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y;

	const int i = threadIdx.y;
	const int j = threadIdx.x;

	const int offset = blockIdx.x * blockDim.x  + blockIdx.y * w*blockDim.y + i*w+j;

	__shared__ float dct_in[65];
	__shared__ float dct_out[65];

	dct_in[i*8+j] = ((float) in_data[offset] - prediction[block_offset + i*8+j]);
	__syncthreads();
	dct_quant_block_8x8(dct_in, dct_out, out_data + block_offset, quant_table + quantization*64, i, j);
}


__device__
static void dequant_idct_block_8x8(float *in_data, float *out_data, const uint8_t* __restrict__ quant_tbl, int i, int j)
{
	// Dequantize
	float dct = in_data[i*8+j];
	out_data[UV_indexes[i*8+j]] = (float) round((dct*quant_tbl[i*8+j]) / 4.0f);
	__syncthreads();

	// Scale
	if(i == 0) {
		out_data[i*8+j] *= ISQRT2;
	}
	if(j == 0) {
		out_data[i*8+j] *= ISQRT2;
	}
	in_data[i*8+j] = out_data[i*8+j];
	__syncthreads();

	// First idct - mb2 = mb - and transpose
	float idct = 0;
	int k;
	#pragma unroll
	for (k = 0; k < 8; ++k) {
		idct += in_data[j*8+k] * dct_lookup[i*8+k];
	}

	out_data[i*8+j] = idct;
	__syncthreads();

	// Second idct - mb2 = mb - and transpose
	idct = 0;
	#pragma unroll
	for (k = 0; k < 8; ++k) {
		idct += out_data[j*8+k] * dct_lookup[i*8+k];
	}
	in_data[i*8+j] = idct;
}

__global__
void dequantize_idct(const int16_t* __restrict__ in_data, const uint8_t* __restrict__ prediction, int w, uint8_t* __restrict__ out_data, int quantization)
{
	const int block_offset = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y;

 	const int i = threadIdx.y;
 	const int j = threadIdx.x;

 	__shared__ float idct_in[65];
 	__shared__ float idct_out[65];

 	idct_in[i*8+j] = (float) in_data[block_offset + i*8+j];
 	__syncthreads();

	dequant_idct_block_8x8(idct_in, idct_out, quant_table + quantization*64, i, j);

	const int offset = blockIdx.x * 8 + blockIdx.y * w*8 + i*w+j;

	int16_t tmp = (int16_t) idct_in[i*8+j] + (int16_t) prediction[block_offset+i*8+j];

	if (tmp < 0)
	{
		tmp = 0;
	}
	else if (tmp > 255)
	{
		tmp = 255;
	}

	out_data[offset] = tmp;
}


static void init_frame_gpu(struct c63_common* cm, struct frame* f)
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

static void deinit_frame_gpu(struct frame* f)
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

struct frame* create_frame(struct c63_common *cm, const struct c63_cuda& c63_cuda)
{
	struct frame *f = (frame*) malloc(sizeof(struct frame));

	f->orig = create_image(cm);
	f->recons = create_image(cm);
	f->predicted = create_image(cm);

	f->residuals = (dct_t*) malloc(sizeof(dct_t));

	dct_t* dct = f->residuals;
	int16_t** residuals[COLOR_COMPONENTS] = { &dct->Ydct, &dct->Udct, &dct->Vdct };

	for (int c = 0; c < COLOR_COMPONENTS; ++c)
	{
		size_t res_size = cm->padw[c] * cm->padh[c] * sizeof(uint16_t);
		int mb_num = cm->mb_cols[c] * cm->mb_rows[c];

		if (on_gpu[c]) {
			cudaMallocHost((void**) residuals[c], res_size);

			cudaMallocHost((void**) &f->mbs[c], mb_num * sizeof(struct macroblock));
			cudaMemsetAsync(f->mbs[c], 0, mb_num * sizeof(struct macroblock), c63_cuda.stream[c]);
		} else {
			*residuals[c] = (int16_t*) malloc(res_size);

			f->mbs[c] = (struct macroblock*) calloc(mb_num, sizeof(struct macroblock));
		}
	}

	init_frame_gpu(cm, f);

	return f;
}

void destroy_frame(struct frame *f)
{
	deinit_frame_gpu(f);

	destroy_image(f->orig);
	destroy_image(f->recons);
	destroy_image(f->predicted);

	dct_t* dct = f->residuals;
	int16_t* residuals[COLOR_COMPONENTS] = { dct->Ydct, dct->Udct, dct->Vdct };

	for (int c = 0; c < COLOR_COMPONENTS; ++c) {
		if (on_gpu[c]) {
			cudaFreeHost(residuals[c]);
			cudaFreeHost(f->mbs[c]);
		} else {
			free(residuals[c]);
			free(f->mbs[c]);
		}
	}

	free(f->residuals);

	free(f);
}

yuv_t* create_image(struct c63_common *cm)
{
	yuv_t* image = (yuv_t*) malloc(sizeof(yuv_t));

	uint8_t** components[COLOR_COMPONENTS] = { &image->Y, &image->U, &image->V };

	for (int c = 0; c < COLOR_COMPONENTS; ++c) {
		size_t size = cm->padw[c] * cm->padh[c] * sizeof(uint8_t);

		if (on_gpu[c]) {
			cudaHostAlloc((void**) components[c], size, cudaHostAllocWriteCombined);
		} else {
			*components[c] = (uint8_t*) malloc(size);
		}
	}

	return image;
}

void destroy_image(yuv_t *image)
{
	uint8_t* components[COLOR_COMPONENTS] = { image->Y, image->U, image->V };

	for (int c = 0; c < COLOR_COMPONENTS; ++c) {
		if (on_gpu[c]) {
			cudaFreeHost(components[c]);
		} else {
			free(components[c]);
		}
	}

	free(image);
}

yuv_t* create_image_gpu(struct c63_common *cm)
{
	yuv_t* image = (yuv_t*) malloc(sizeof(yuv_t));
	cudaMalloc((void**) &image->Y, cm->ypw * cm->yph * sizeof(uint8_t));
	cudaMalloc((void**) &image->U, cm->upw * cm->uph * sizeof(uint8_t));
	cudaMalloc((void**) &image->V, cm->vpw * cm->vph * sizeof(uint8_t));

	return image;
}

void destroy_image_gpu(yuv_t* image)
{
	cudaFree(image->Y);
	cudaFree(image->U);
	cudaFree(image->V);
	free(image);
}

void dump_image(yuv_t *image, int w, int h, FILE *fp)
{
	fwrite(image->Y, 1, w * h, fp);
	fwrite(image->U, 1, w * h / 4, fp);
	fwrite(image->V, 1, w * h / 4, fp);
}
