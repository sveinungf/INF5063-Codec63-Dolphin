#ifndef ALLOCATION_H_
#define ALLOCATION_H_

#include "c63.h"


void c63_motion_estimate_gpu(struct c63_common* cm, const struct c63_common_gpu& cm_gpu,
		const struct c63_cuda& c63_cuda);
void c63_motion_estimate_host(struct c63_common* cm);

void c63_motion_compensate_gpu(struct c63_common *cm, const struct c63_cuda& c63_cuda);
void c63_motion_compensate_host(struct c63_common* cm);

void zero_out_prediction_gpu(struct c63_common* cm, const struct c63_cuda& c63_cuda);
void zero_out_prediction_host(struct c63_common* cm);

void dct_quantize_gpu(struct c63_common* cm, const struct c63_cuda& c63_cuda);
void dct_quantize_host(struct c63_common* cm);

void dequantize_idct_gpu(struct c63_common* cm, const struct c63_cuda& c63_cuda);
void dequantize_idct_host(struct c63_common* cm);

#endif /* ALLOCATION_H_ */
