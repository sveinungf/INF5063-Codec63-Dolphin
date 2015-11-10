#ifndef C63_DSP_SIMD_H_
#define C63_DSP_SIMD_H_

#define ISQRT2 0.70710678118654f

#include <inttypes.h>

void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data, uint8_t *quant_tbl);

void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data, uint8_t *quant_tbl);

#endif  /* C63_DSP_SIMD_H_ */
