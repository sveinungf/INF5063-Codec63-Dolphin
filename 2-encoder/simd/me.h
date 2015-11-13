#ifndef C63_ME_SIMD_H_
#define C63_ME_SIMD_H_

#include "c63.h"


void c63_motion_estimate(struct c63_common *cm, int component);

void c63_motion_compensate(struct c63_common *cm, int component);

#endif  /* C63_ME_SIMD_H_ */
