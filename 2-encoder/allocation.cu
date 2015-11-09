#include "../common/sisci_common.h"
#include "allocation.h"
#include "cuda/me.h"

extern "C" {
#include "simd/me.h"
}

static const int Y = Y_COMPONENT;
static const int U = U_COMPONENT;
static const int V = V_COMPONENT;

template<int component>
static void c63_motion_estimate_host(struct c63_common* cm)
{
	int w = cm->padw[component];
	int h = cm->padh[component];
	int cols = cm->mb_cols[component];
	int rows = cm->mb_rows[component];
	struct macroblock* mb = cm->curframe->mbs[component];
	struct macroblock* mb_gpu = cm->curframe->mbs_gpu[component];

	uint8_t* orig;
	uint8_t* orig_gpu;
	uint8_t* recons;
	uint8_t* recons_gpu;

	switch (component)
	{
		case Y_COMPONENT:
			orig = cm->curframe->orig->Y;
			orig_gpu = (uint8_t*) cm->curframe->orig_gpu->Y;
			recons = cm->refframe->recons->Y;
			recons_gpu = cm->refframe->recons_gpu->Y;
			break;
		case U_COMPONENT:
			orig = cm->curframe->orig->U;
			orig_gpu = (uint8_t*) cm->curframe->orig_gpu->U;
			recons = cm->refframe->recons->U;
			recons_gpu = cm->refframe->recons_gpu->U;
			break;
		case V_COMPONENT:
			orig = cm->curframe->orig->V;
			orig_gpu = (uint8_t*) cm->curframe->orig_gpu->V;
			recons = cm->refframe->recons->V;
			recons_gpu = cm->refframe->recons_gpu->V;
			break;
	}

	cudaMemcpy(orig, orig_gpu, w * h * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(recons, recons_gpu, w * h * sizeof(uint8_t), cudaMemcpyDeviceToHost);

	c63_motion_estimate(cm, component);

	cudaMemcpy(mb_gpu, mb, cols * rows * sizeof(struct macroblock), cudaMemcpyHostToDevice);
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
	c63_motion_estimate_host<Y>(cm);
#endif
#if !(U_ON_GPU)
	c63_motion_estimate_host<U>(cm);
#endif
#if !(V_ON_GPU)
	c63_motion_estimate_host<V>(cm);
#endif
}
