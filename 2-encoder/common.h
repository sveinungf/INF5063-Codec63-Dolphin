#ifndef C63_COMMON_H_
#define C63_COMMON_H_

#include <inttypes.h>

#include "c63.h"
#include "c63_cuda.h"

__constant__ uint8_t quant_table[192] =
{
	6,  4,  4,  5,  4,  4,  6,  5,
	5,  5,  7,  6,  6,  7,  9,  16,
	10,  9,  8,  8,  9,  19,  14,  14,
	11,  16,  23,  20,  24,  12,  22,  20,
	22,  22,  25,  28,  36,  31,  25,  27,
	34,  27,  22,  22,  32,  43,  32,  34,
	38,  39,  41,  41,  41,  24,  30,  45,
	48,  44,  40,  48,  36,  40,  41,  39,
	/*************************************/
	6,  7,  7,  9,  8,  9,  18,  10,
	10,  18,  39,  26,  22,  26,  39,  39,
	39,  39,  39,  39,  39,  39,  39,  39,
	39,  39,  39,  39,  39,  39,  39,  39,
	39,  39,  39,  39,  39,  39,  39,  39,
	39,  39,  39,  39,  39,  39,  39,  39,
	39,  39,  39,  39,  39,  39,  39,  39,
	39,  39,  39,  39,  39,  39,  39,  39,
	/*************************************/
	6,  7,  7,  9,  8,  9,  18,  10,
	10,  18,  39,  26,  22,  26,  39,  39,
	39,  39,  39,  39,  39,  39,  39,  39,
	39,  39,  39,  39,  39,  39,  39,  39,
	39,  39,  39,  39,  39,  39,  39,  39,
	39,  39,  39,  39,  39,  39,  39,  39,
	39,  39,  39,  39,  39,  39,  39,  39,
	39,  39,  39,  39,  39,  39,  39,  39
};


__constant__ float dct_lookup[64] =
{
  1.0f,  0.980785f,  0.923880f,  0.831470f,  0.707107f,  0.555570f,  0.382683f,  0.195090f,
  1.0f,  0.831470f,  0.382683f, -0.195090f, -0.707107f, -0.980785f, -0.923880f, -0.555570f,
  1.0f,  0.555570f, -0.382683f, -0.980785f, -0.707107f,  0.195090f,  0.923880f,  0.831470f,
  1.0f,  0.195090f, -0.923880f, -0.555570f,  0.707107f,  0.831470f, -0.382683f, -0.980785f,
  1.0f, -0.195090f, -0.923880f,  0.555570f,  0.707107f, -0.831470f, -0.382683f,  0.980785f,
  1.0f, -0.555570f, -0.382683f,  0.980785f, -0.707107f, -0.195090f,  0.923880f, -0.831470f,
  1.0f, -0.831470f,  0.382683f,  0.195090f, -0.707107f,  0.980785f, -0.923880f,  0.555570f,
  1.0f, -0.980785f,  0.923880f, -0.831470f,  0.707107f, -0.555570f,  0.382683f, -0.195090f
};

__constant__ float dct_lookup_trans[64] =
{
  1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f,  1.000000f,
  0.980785f,  0.831470f,  0.555570f,  0.195090f, -0.195090f, -0.555570f, -0.831470f, -0.980785f,
  0.923880f,  0.382683f, -0.382683f, -0.923880f, -0.923880f, -0.382683f,  0.382683f,  0.923880f,
  0.831470f, -0.195090f, -0.980785f, -0.555570f,  0.555570f,  0.980785f,  0.195090f, -0.831470f,
  0.707107f, -0.707107f, -0.707107f,  0.707107f,  0.707107f, -0.707107f, -0.707107f,  0.707107f,
  0.555570f, -0.980785f,  0.195090f,  0.831470f, -0.831470f, -0.195090f,  0.980785f, -0.555570f,
  0.382683f, -0.923880f,  0.923880f, -0.382683f, -0.382683f,  0.923880f, -0.923880f,  0.382683f,
  0.195090f, -0.555570f,  0.831470f, -0.980785f,  0.980785f, -0.831470f,  0.555570f, -0.195090f
};


/* Array containing the indexes resulting from calculating
 * (zigzag_V[zigzag]*8) + zigzag_U[zigzag] for zigzag = 0, 1, ..., 63
 */
__constant__ uint8_t UV_indexes[64] =
{
	 0,  1,  8, 16,  9,  2,  3, 10,
	17, 24, 32, 25, 18, 11,  4,  5,
	12, 19, 26, 33, 40, 48, 41, 34,
	27, 20, 13,  6,  7, 14, 21, 28,
	35, 42, 49, 56, 57, 50, 43, 36,
	29, 22, 15, 23, 30, 37, 44, 51,
	58, 59, 52, 45, 38, 31, 39, 46,
	53, 60, 61, 54, 47, 55, 62, 63,
};

// Declarations
struct frame* create_frame(struct c63_common *cm, const struct c63_cuda& c63_cuda);

yuv_t* create_image(struct c63_common *cm);

yuv_t* create_image_gpu(struct c63_common *cm);

__global__
void dct_quantize(const uint8_t* __restrict__ in_data, const uint8_t* __restrict__ prediction, int w, int16_t* __restrict__ out_data, int quantization);

__global__
void dequantize_idct(const int16_t* __restrict__ in_data, const uint8_t* __restrict__ prediction, int w, uint8_t* __restrict__ out_data, int quantization);

void destroy_frame(struct frame *f);

void destroy_image(yuv_t* image);

void destroy_image_gpu(yuv_t* image);

void dump_image(yuv_t *image, int w, int h, FILE *fp);

void cuda_init(struct c63_common cm);
void cuda_cleanup();

#endif  /* C63_COMMON_H_ */
