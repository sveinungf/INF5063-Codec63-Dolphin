#ifndef ALLOCATION_H_
#define ALLOCATION_H_

#include "c63.h"

#define Y_ON_GPU 0
#define U_ON_GPU 0
#define V_ON_GPU 0

void c63_motion_estimate_gpu(struct c63_common* cm, const struct c63_common_gpu& cm_gpu,
		const struct c63_cuda& c63_cuda);
void c63_motion_estimate_host(struct c63_common* cm);

void c63_motion_compensate_host(struct c63_common* cm);

void dct_quantize_host(struct c63_common* cm);

void dequantize_idct_host(struct c63_common* cm);

#endif /* ALLOCATION_H_ */
