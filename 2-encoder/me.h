#ifndef C63_ME_H_
#define C63_ME_H_

#include "c63.h"

// Declaration

void gpu_c63_motion_estimate(struct c63_common *cm, const struct c63_common_gpu& cm_gpu,
		const struct c63_cuda& c63_cuda);

void gpu_c63_motion_compensate(struct c63_common *cm, const struct c63_cuda& c63_cuda);

#endif  /* C63_ME_H_ */
