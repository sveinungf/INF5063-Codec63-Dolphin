#ifndef ALLOCATION_H_
#define ALLOCATION_H_

#include "c63.h"

#define Y_ON_GPU 1
#define U_ON_GPU 1
#define V_ON_GPU 1

void c63_motion_estimate_gpu(struct c63_common* cm, const struct c63_common_gpu& cm_gpu,
		const struct c63_cuda& c63_cuda);
void c63_motion_estimate_host(struct c63_common* cm);

#endif /* ALLOCATION_H_ */
