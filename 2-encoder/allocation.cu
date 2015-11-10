#include "../common/sisci_common.h"
#include "allocation.h"
#include "cuda/me.h"

extern "C" {
#include "simd/common.h"
#include "simd/me.h"
}


static const int Y = Y_COMPONENT;
static const int U = U_COMPONENT;
static const int V = V_COMPONENT;

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

	dct_quantize_host(orig, predicted, w, h, residuals, quanttbl);
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

	dequantize_idct_host(residuals, predicted, w, h, recons, quanttbl);
}

void c63_motion_estimate_gpu(struct c63_common* cm, const struct c63_common_gpu& cm_gpu,
		const struct c63_cuda& c63_cuda)
{
#if Y_ON_GPU
	gpu_c63_motion_estimate<Y>(cm, cm_gpu, c63_cuda);
#endif
#if U_ON_GPU
	gpu_c63_motion_estimate<U>(cm, cm_gpu, c63_cuda);
#endif
#if V_ON_GPU
	gpu_c63_motion_estimate<V>(cm, cm_gpu, c63_cuda);
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
