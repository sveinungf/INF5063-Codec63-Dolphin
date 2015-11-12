#include "../common/sisci_common.h"
#include "allocation.h"
#include "cuda/dsp.h"
#include "cuda/me.h"

extern "C" {
#include "simd/dsp.h"
#include "simd/me.h"
}

using namespace c63;

static const int Y = Y_COMPONENT;
static const int U = U_COMPONENT;
static const int V = V_COMPONENT;

template<int component>
static inline void zero_out_prediction_gpu(struct c63_common* cm, const struct c63_cuda& c63_cuda)
{
	const int w = cm->padw[component];
	const int h = cm->padh[component];
	const cudaStream_t stream = c63_cuda.stream[component];

	uint8_t* predicted;

	switch (component)
	{
		case Y_COMPONENT:
			predicted = cm->curframe->predicted_gpu->Y;
			break;
		case U_COMPONENT:
			predicted = cm->curframe->predicted_gpu->U;
			break;
		case V_COMPONENT:
			predicted = cm->curframe->predicted_gpu->V;
			break;
	}

	cudaMemsetAsync(predicted, 0, w * h * sizeof(uint8_t), stream);
}

template<int component>
static inline void zero_out_prediction_host(struct c63_common* cm)
{
	const int w = cm->padw[component];
	const int h = cm->padh[component];

	uint8_t* predicted;

	switch (component)
	{
		case Y_COMPONENT:
			predicted = cm->curframe->predicted->Y;
			break;
		case U_COMPONENT:
			predicted = cm->curframe->predicted->U;
			break;
		case V_COMPONENT:
			predicted = cm->curframe->predicted->V;
			break;
	}

	memset(predicted, 0, w * h * sizeof(uint8_t));
}

template<int component>
static inline void dct_quantize_gpu(struct c63_common* cm, const struct c63_cuda& c63_cuda)
{
	const int w = cm->padw[component];
	const int h = cm->padh[component];
	const cudaStream_t stream = c63_cuda.stream[component];

	uint8_t* orig;
	uint8_t* predicted;
	int16_t* residuals;
	int16_t* residuals_host;

	struct frame* f = cm->curframe;

	switch (component)
	{
		case Y_COMPONENT:
			orig = (uint8_t*) f->orig_gpu->Y;
			predicted = f->predicted_gpu->Y;
			residuals = f->residuals_gpu->Ydct;
			residuals_host = f->residuals->Ydct;
			break;
		case U_COMPONENT:
			orig = (uint8_t*) f->orig_gpu->U;
			predicted = f->predicted_gpu->U;
			residuals = f->residuals_gpu->Udct;
			residuals_host = f->residuals->Udct;
			break;
		case V_COMPONENT:
			orig = (uint8_t*) f->orig_gpu->V;
			predicted = f->predicted_gpu->V;
			residuals = f->residuals_gpu->Vdct;
			residuals_host = f->residuals->Vdct;
			break;
	}

	const dim3 threadsPerBlock(8, 8);
	const dim3 numBlocks(w / threadsPerBlock.x, h / threadsPerBlock.y);

	dct_quantize<<<numBlocks, threadsPerBlock, 0, stream>>>(orig, predicted, w, residuals,
			component);
	cudaMemcpyAsync(residuals_host, residuals, w * h * sizeof(int16_t), cudaMemcpyDeviceToHost,
			stream);
}

template<int component>
static inline void dct_quantize_host(struct c63_common* cm)
{
	const int w = cm->padw[component];
	const int h = cm->padh[component];
	uint8_t* quanttbl = cm->quanttbl[component];

	uint8_t* orig;
	uint8_t* predicted;
	int16_t* residuals;

	switch (component)
	{
		case Y_COMPONENT:
			orig = cm->curframe->orig->Y;
			predicted = cm->curframe->predicted->Y;
			residuals = cm->curframe->residuals->Ydct;
			break;
		case U_COMPONENT:
			orig = cm->curframe->orig->U;
			predicted = cm->curframe->predicted->U;
			residuals = cm->curframe->residuals->Udct;
			break;
		case V_COMPONENT:
			orig = cm->curframe->orig->V;
			predicted = cm->curframe->predicted->V;
			residuals = cm->curframe->residuals->Vdct;
			break;
	}

	dct_quantize(orig, predicted, w, h, residuals, quanttbl);
}

template<int component>
static inline void dequantize_idct_gpu(struct c63_common* cm, const struct c63_cuda& c63_cuda)
{
	const int w = cm->padw[component];
	const int h = cm->padh[component];
	const cudaStream_t stream = c63_cuda.stream[component];

	uint8_t* predicted;
	uint8_t* recons;
	int16_t* residuals;

	struct frame* f = cm->curframe;

	switch (component)
	{
		case Y_COMPONENT:
			predicted = f->predicted_gpu->Y;
			recons = f->recons_gpu->Y;
			residuals = f->residuals_gpu->Ydct;
			break;
		case U_COMPONENT:
			predicted = f->predicted_gpu->U;
			recons = f->recons_gpu->U;
			residuals = f->residuals_gpu->Udct;
			break;
		case V_COMPONENT:
			predicted = f->predicted_gpu->V;
			recons = f->recons_gpu->V;
			residuals = f->residuals_gpu->Vdct;
			break;
	}

	const dim3 threadsPerBlock(8, 8);
	const dim3 numBlocks(w / threadsPerBlock.x, h / threadsPerBlock.y);

	dequantize_idct<<<numBlocks, threadsPerBlock, 0, stream>>>(residuals, predicted, w, recons,
			component);
}

template<int component>
static inline void dequantize_idct_host(struct c63_common* cm)
{
	const int w = cm->padw[component];
	const int h = cm->padh[component];
	uint8_t* quanttbl = cm->quanttbl[component];

	uint8_t* predicted;
	uint8_t* recons;
	int16_t* residuals;

	switch (component)
	{
		case Y_COMPONENT:
			predicted = cm->curframe->predicted->Y;
			recons = cm->curframe->recons->Y;
			residuals = cm->curframe->residuals->Ydct;
			break;
		case U_COMPONENT:
			predicted = cm->curframe->predicted->U;
			recons = cm->curframe->recons->U;
			residuals = cm->curframe->residuals->Udct;
			break;
		case V_COMPONENT:
			predicted = cm->curframe->predicted->V;
			recons = cm->curframe->recons->V;
			residuals = cm->curframe->residuals->Vdct;
			break;
	}

	dequantize_idct(residuals, predicted, w, h, recons, quanttbl);
}

void c63_motion_estimate_gpu(struct c63_common* cm, const struct c63_common_gpu& cm_gpu,
		const struct c63_cuda& c63_cuda)
{
#if Y_ON_GPU
	gpu::c63_motion_estimate<Y>(cm, cm_gpu, c63_cuda);
#endif
#if U_ON_GPU
	gpu::c63_motion_estimate<U>(cm, cm_gpu, c63_cuda);
#endif
#if V_ON_GPU
	gpu::c63_motion_estimate<V>(cm, cm_gpu, c63_cuda);
#endif
}

void c63_motion_estimate_host(struct c63_common* cm)
{
#if !(Y_ON_GPU)
	c63_motion_estimate(cm, Y);
#endif
#if !(U_ON_GPU)
	c63_motion_estimate(cm, U);
#endif
#if !(V_ON_GPU)
	c63_motion_estimate(cm, V);
#endif
}

void c63_motion_compensate_gpu(struct c63_common *cm, const struct c63_cuda& c63_cuda)
{
#if Y_ON_GPU
	gpu::c63_motion_compensate<Y>(cm, c63_cuda);
#endif
#if U_ON_GPU
	gpu::c63_motion_compensate<U>(cm, c63_cuda);
#endif
#if V_ON_GPU
	gpu::c63_motion_compensate<V>(cm, c63_cuda);
#endif
}

void c63_motion_compensate_host(struct c63_common* cm)
{
#if !(Y_ON_GPU)
	c63_motion_compensate(cm, Y);
#endif
#if !(U_ON_GPU)
	c63_motion_compensate(cm, U);
#endif
#if !(V_ON_GPU)
	c63_motion_compensate(cm, V);
#endif
}

void zero_out_prediction_gpu(struct c63_common* cm, const struct c63_cuda& c63_cuda)
{
#if Y_ON_GPU
	zero_out_prediction_gpu<Y>(cm, c63_cuda);
#endif
#if U_ON_GPU
	zero_out_prediction_gpu<U>(cm, c63_cuda);
#endif
#if V_ON_GPU
	zero_out_prediction_gpu<V>(cm, c63_cuda);
#endif
}

void zero_out_prediction_host(struct c63_common* cm)
{
#if !(Y_ON_GPU)
	zero_out_prediction_host<Y>(cm);
#endif
#if !(U_ON_GPU)
	zero_out_prediction_host<U>(cm);
#endif
#if !(V_ON_GPU)
	zero_out_prediction_host<V>(cm);
#endif
}

void dct_quantize_gpu(struct c63_common* cm, const struct c63_cuda& c63_cuda)
{
#if Y_ON_GPU
	dct_quantize_gpu<Y>(cm, c63_cuda);
#endif
#if U_ON_GPU
	dct_quantize_gpu<U>(cm, c63_cuda);
#endif
#if V_ON_GPU
	dct_quantize_gpu<V>(cm, c63_cuda);
#endif
}

void dct_quantize_host(struct c63_common* cm)
{
#if !(Y_ON_GPU)
	dct_quantize_host<Y>(cm);
#endif
#if !(U_ON_GPU)
	dct_quantize_host<U>(cm);
#endif
#if !(V_ON_GPU)
	dct_quantize_host<V>(cm);
#endif
}

void dequantize_idct_gpu(struct c63_common* cm, const struct c63_cuda& c63_cuda)
{
#if Y_ON_GPU
	dequantize_idct_gpu<Y>(cm, c63_cuda);
#endif
#if U_ON_GPU
	dequantize_idct_gpu<U>(cm, c63_cuda);
#endif
#if V_ON_GPU
	dequantize_idct_gpu<V>(cm, c63_cuda);
#endif
}

void dequantize_idct_host(struct c63_common* cm)
{
#if !(Y_ON_GPU)
	dequantize_idct_host<Y>(cm);
#endif
#if !(U_ON_GPU)
	dequantize_idct_host<U>(cm);
#endif
#if !(V_ON_GPU)
	dequantize_idct_host<V>(cm);
#endif
}
