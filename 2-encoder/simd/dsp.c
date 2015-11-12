#include <immintrin.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "../../common/tables.h"
#include "dsp.h"


static void transpose_block(float *in_data, float *out_data)
{
	int i;

	__m128 row1, row2, row3, row4;

	for (i = 0; i < 8; i += 4)
	{
		/* Transpose one 4x8 matrix at a time by using _MM_TRANSPOSE4_PS
		 * on two 4x4 matrixes
		 * First iteration: upper left and lower left
		 * Second iteration: upper right and lower right
		 */

		// Transpose the upper 4x4 matrix
		row1 = _mm_load_ps(in_data + i);
		row2 = _mm_load_ps(in_data + 8 + i);
		row3 = _mm_load_ps(in_data + 16 + i);
		row4 = _mm_load_ps(in_data + 24 + i);
		_MM_TRANSPOSE4_PS(row1, row2, row3, row4);

		// Store the first four elements of each row of the transposed 8x8 matrix
		_mm_store_ps(out_data + i * 8, row1);
		_mm_store_ps(out_data + (i + 1) * 8, row2);
		_mm_store_ps(out_data + (i + 2) * 8, row3);
		_mm_store_ps(out_data + (i + 3) * 8, row4);

		// Transpose the lower 4x4 matrix
		row1 = _mm_load_ps(in_data + 32 + i);
		row2 = _mm_load_ps(in_data + 40 + i);
		row3 = _mm_load_ps(in_data + 48 + i);
		row4 = _mm_load_ps(in_data + 56 + i);
		_MM_TRANSPOSE4_PS(row1, row2, row3, row4);

		// Store the last four elements of each row of the transposed 8x8 matrix
		_mm_store_ps(out_data + i * 8 + 4, row1);
		_mm_store_ps(out_data + (i + 1) * 8 + 4, row2);
		_mm_store_ps(out_data + (i + 2) * 8 + 4, row3);
		_mm_store_ps(out_data + (i + 3) * 8 + 4, row4);
	}
}

static void dct_1d_general(float* in_data, float* out_data, float lookup[64])
{
	__m256 current, dct_values, multiplied, sum;

	current = _mm256_broadcast_ss(in_data);
	dct_values = _mm256_load_ps(lookup);
	multiplied = _mm256_mul_ps(dct_values, current);
	sum = multiplied;

	// Broadcasts a single float (scalar) to every element in 'current'.
	current = _mm256_broadcast_ss(in_data + 1);
	// Loads DCT values from the lookup table. iDCT uses a transposed lookup table here.
	dct_values = _mm256_load_ps(lookup + 8);
	// Vertically multiply the scalar with the DCT values.
	multiplied = _mm256_mul_ps(dct_values, current);
	// Vertically add to the previous sum.
	sum = _mm256_add_ps(sum, multiplied);

	current = _mm256_broadcast_ss(in_data + 2);
	dct_values = _mm256_load_ps(lookup + 16);
	multiplied = _mm256_mul_ps(dct_values, current);
	sum = _mm256_add_ps(sum, multiplied);

	current = _mm256_broadcast_ss(in_data + 3);
	dct_values = _mm256_load_ps(lookup + 24);
	multiplied = _mm256_mul_ps(dct_values, current);
	sum = _mm256_add_ps(sum, multiplied);

	current = _mm256_broadcast_ss(in_data + 4);
	dct_values = _mm256_load_ps(lookup + 32);
	multiplied = _mm256_mul_ps(dct_values, current);
	sum = _mm256_add_ps(sum, multiplied);

	current = _mm256_broadcast_ss(in_data + 5);
	dct_values = _mm256_load_ps(lookup + 40);
	multiplied = _mm256_mul_ps(dct_values, current);
	sum = _mm256_add_ps(sum, multiplied);

	current = _mm256_broadcast_ss(in_data + 6);
	dct_values = _mm256_load_ps(lookup + 48);
	multiplied = _mm256_mul_ps(dct_values, current);
	sum = _mm256_add_ps(sum, multiplied);

	current = _mm256_broadcast_ss(in_data + 7);
	dct_values = _mm256_load_ps(lookup + 56);
	multiplied = _mm256_mul_ps(dct_values, current);
	sum = _mm256_add_ps(sum, multiplied);

	_mm256_store_ps(out_data, sum);
}

static void scale_block(float *in_data, float *out_data)
{
	__m256 in_vector, result;

	// Load the a1 values into a register
	static float a1_values[8] __attribute__((aligned(32))) = { ISQRT2, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			1.0f, 1.0f };
	__m256 a1 = _mm256_load_ps(a1_values);

	// Load the a2 values into a register for the exception case
	__m256 a2 = _mm256_set1_ps(ISQRT2);

	/* First row is an exception
	 * Requires two _mm256_mul_ps operations */
	in_vector = _mm256_load_ps(in_data);
	result = _mm256_mul_ps(in_vector, a1);
	result = _mm256_mul_ps(result, a2);
	_mm256_store_ps(out_data, result);

	// Remaining calculations can be done with one _mm256_mul_ps operation
	in_vector = _mm256_load_ps(in_data + 8);
	result = _mm256_mul_ps(in_vector, a1);
	_mm256_store_ps(out_data + 8, result);

	in_vector = _mm256_load_ps(in_data + 16);
	result = _mm256_mul_ps(in_vector, a1);
	_mm256_store_ps(out_data + 16, result);

	in_vector = _mm256_load_ps(in_data + 24);
	result = _mm256_mul_ps(in_vector, a1);
	_mm256_store_ps(out_data + 24, result);

	in_vector = _mm256_load_ps(in_data + 32);
	result = _mm256_mul_ps(in_vector, a1);
	_mm256_store_ps(out_data + 32, result);

	in_vector = _mm256_load_ps(in_data + 40);
	result = _mm256_mul_ps(in_vector, a1);
	_mm256_store_ps(out_data + 40, result);

	in_vector = _mm256_load_ps(in_data + 48);
	result = _mm256_mul_ps(in_vector, a1);
	_mm256_store_ps(out_data + 48, result);

	in_vector = _mm256_load_ps(in_data + 56);
	result = _mm256_mul_ps(in_vector, a1);
	_mm256_store_ps(out_data + 56, result);
}

// Rounding half away from zero (equivalent to round() from math.h)
// __m256 contains 8 floats, but to simplify the examples, only 4 will be shown
// Initial values to be used in the examples:
// [-12.49  -0.5   1.5   3.7]
static __m256 c63_mm256_roundhalfawayfromzero_ps(const __m256 initial)
{
	const __m256 sign_mask = _mm256_set1_ps(-0.f);
	const __m256 one_half = _mm256_set1_ps(0.5f);
	const __m256 all_zeros = _mm256_setzero_ps();
	const __m256 pos_one = _mm256_set1_ps(1.f);
	const __m256 neg_one = _mm256_set1_ps(-1.f);

	// Creates a mask based on the sign of the floats, true for negative floats
	// Example: [true   true   false   false]
	__m256 less_than_zero = _mm256_cmp_ps(initial, all_zeros, _CMP_LT_OQ);

	// Returns the integer part of the floats
	// Example: [-12.0   -0.0   1.0   3.0]
	__m256 without_fraction = _mm256_round_ps(initial, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));

	// Returns the fraction part of the floats
	// Example: [-0.49   -0.5   0.5   0.7]
	__m256 fraction = _mm256_sub_ps(initial, without_fraction);

	// Absolute values of the fractions
	// Example: [0.49   0.5   0.5   0.7]
	__m256 fraction_abs = _mm256_andnot_ps(sign_mask, fraction);

	// Compares abs(fractions) to 0.5, true if lower
	// Example: [true   false   false   false]
	__m256 less_than_one_half = _mm256_cmp_ps(fraction_abs, one_half, _CMP_LT_OQ);

	// Blends 1.0 and -1.0 depending on the initial sign of the floats
	// Example: [-1.0   -1.0   1.0   1.0]
	__m256 signed_ones = _mm256_blendv_ps(pos_one, neg_one, less_than_zero);

	// Blends the previous result with zeros depending on the fractions that are lower than 0.5
	// Example: [0.0   -1.0   1.0   1.0]
	__m256 to_add = _mm256_blendv_ps(signed_ones, all_zeros, less_than_one_half);

	// Adds the previous result to the floats without fractions
	// Example: [-12.0   -1.0   2.0   4.0]
	return _mm256_add_ps(without_fraction, to_add);
}

static void quantize_block(float *in_data, float *out_data, uint8_t *quant_tbl)
{
	int zigzag;

	__m128i quants;
	__m128 quant_lo, quant_hi;

	__m256 result, dct_values, quant_values;
	__m256 factor = _mm256_set1_ps(0.25f);

	for (zigzag = 0; zigzag < 64; zigzag += 8)
	{
		// Set the dct_values for the current interation
		dct_values = _mm256_set_ps(in_data[UV_indexes[zigzag + 7]],
				in_data[UV_indexes[zigzag + 6]], in_data[UV_indexes[zigzag + 5]],
				in_data[UV_indexes[zigzag + 4]], in_data[UV_indexes[zigzag + 3]],
				in_data[UV_indexes[zigzag + 2]], in_data[UV_indexes[zigzag + 1]],
				in_data[UV_indexes[zigzag]]);

		// Multiply with 0.25 to divide by 4.0
		result = _mm256_mul_ps(dct_values, factor);

		/* Load values from quant_tbl, extract the eight first values as 32-bit integers
		 * and convert them to floating-point values */
		quants = _mm_loadl_epi64((__m128i *) &quant_tbl[zigzag]);
		quant_lo = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(quants));
		quant_hi = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi32(quants, 0b00000001)));

		// Combine the two 128-bit registers containing quant-values and multiply with previous product
		quant_values = _mm256_insertf128_ps(_mm256_castps128_ps256(quant_lo), quant_hi, 0b00000001);
		result = _mm256_div_ps(result, quant_values);

		// Round off values and store in out_data buffer
		result = c63_mm256_roundhalfawayfromzero_ps(result);
		_mm256_store_ps(out_data + zigzag, result);
	}
}

static void dequantize_block(float *in_data, float *out_data, uint8_t *quant_tbl)
{
	int zigzag;

	// Temporary buffer
	float temp_buf[8] __attribute__((aligned(32)));

	__m128i quants;
	__m128 quant_lo, quant_hi;

	__m256 result, dct_values, quant_values;
	__m256 factor = _mm256_set1_ps(0.25f);

	for (zigzag = 0; zigzag < 64; zigzag += 8)
	{
		// Load dct-values
		dct_values = _mm256_load_ps(in_data + zigzag);

		/* Load values from quant_tbl, extract the eight first values as 32-bit integers
		 * and convert them to floating-point values */
		quants = _mm_loadl_epi64((__m128i *) &quant_tbl[zigzag]);
		quant_lo = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(quants));
		quant_hi = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi32(quants, 0b00000001)));

		/* Combine the two 128-bit registers containing quant-values
		 * and multiply with the register containing dct-values */
		quant_values = _mm256_insertf128_ps(_mm256_castps128_ps256(quant_lo), quant_hi, 0b00000001);
		result = _mm256_mul_ps(dct_values, quant_values);

		// Multiply with 0.25 to divide by 4.0
		result = _mm256_mul_ps(result, factor);

		// Round off products and store them temporarily
		result = c63_mm256_roundhalfawayfromzero_ps(result);
		_mm256_store_ps(temp_buf, result);

		// Store the results at the correct places in the out_data buffer
		out_data[UV_indexes[zigzag]] = temp_buf[0];
		out_data[UV_indexes[zigzag + 1]] = temp_buf[1];
		out_data[UV_indexes[zigzag + 2]] = temp_buf[2];
		out_data[UV_indexes[zigzag + 3]] = temp_buf[3];
		out_data[UV_indexes[zigzag + 4]] = temp_buf[4];
		out_data[UV_indexes[zigzag + 5]] = temp_buf[5];
		out_data[UV_indexes[zigzag + 6]] = temp_buf[6];
		out_data[UV_indexes[zigzag + 7]] = temp_buf[7];
	}
}

static void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data, uint8_t *quant_tbl)
{
	float mb[8 * 8] __attribute((aligned(32)));
	float mb2[8 * 8] __attribute((aligned(32)));

	int i, v;

	for (i = 0; i < 64; ++i)
	{
		mb2[i] = in_data[i];
	}

	/* Two 1D DCT operations with transpose */
	for (v = 0; v < 8; ++v)
	{
		dct_1d_general(mb2 + v * 8, mb + v * 8, dctlookup);
	}

	transpose_block(mb, mb2);

	for (v = 0; v < 8; ++v)
	{
		dct_1d_general(mb2 + v * 8, mb + v * 8, dctlookup);
	}
	transpose_block(mb, mb2);

	scale_block(mb2, mb);
	quantize_block(mb, mb2, quant_tbl);

	for (i = 0; i < 64; ++i)
	{
		out_data[i] = mb2[i];
	}
}

static void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data, uint8_t *quant_tbl)
{
	float mb[8 * 8] __attribute((aligned(32)));
	float mb2[8 * 8] __attribute((aligned(32)));

	int i, v;

	for (i = 0; i < 64; ++i)
	{
		mb[i] = in_data[i];
	}

	dequantize_block(mb, mb2, quant_tbl);
	scale_block(mb2, mb);

	/* Two 1D inverse DCT operations with transpose */
	for (v = 0; v < 8; ++v)
	{
		dct_1d_general(mb + v * 8, mb2 + v * 8, dctlookup_trans);
	}

	transpose_block(mb2, mb);

	for (v = 0; v < 8; ++v)
	{
		dct_1d_general(mb + v * 8, mb2 + v * 8, dctlookup_trans);
	}

	transpose_block(mb2, mb);

	for (i = 0; i < 64; ++i)
	{
		out_data[i] = mb[i];
	}
}

static void dequantize_idct_row(int16_t *in_data, uint8_t *prediction, int w, int h, int y,
		uint8_t *out_data, uint8_t *quantization)
{
	int x;

	int16_t block[8 * 8];

	/* Perform the dequantization and iDCT */
	for (x = 0; x < w; x += 8)
	{
		int i, j;

		dequant_idct_block_8x8(in_data + (x * 8), block, quantization);

		for (i = 0; i < 8; ++i)
		{
			for (j = 0; j < 8; ++j)
			{
				/* Add prediction block. Note: DCT is not precise -
				 Clamp to legal values */
				int16_t tmp = block[i * 8 + j] + (int16_t) prediction[i * w + j + x];

				if (tmp < 0)
				{
					tmp = 0;
				}
				else if (tmp > 255)
				{
					tmp = 255;
				}

				out_data[i * w + j + x] = tmp;
			}
		}
	}
}

static void dct_quantize_row(uint8_t *in_data, uint8_t *prediction, int w, int h, int16_t *out_data,
		uint8_t *quantization)
{
	int x;

	int16_t block[8 * 8];

	/* Perform the DCT and quantization */
	for (x = 0; x < w; x += 8)
	{
		int i, j;

		for (i = 0; i < 8; ++i)
		{
			for (j = 0; j < 8; ++j)
			{
				block[i * 8 + j] = ((int16_t) in_data[i * w + j + x] - prediction[i * w + j + x]);
			}
		}

		/* Store MBs linear in memory, i.e. the 64 coefficients are stored
		 continous. This allows us to ignore stride in DCT/iDCT and other
		 functions. */
		dct_quant_block_8x8(block, out_data + (x * 8), quantization);
	}
}

void dequantize_idct(int16_t *in_data, uint8_t *prediction, uint32_t width, uint32_t height,
		uint8_t *out_data, uint8_t *quantization)
{
	int y;

	for (y = 0; y < height; y += 8)
	{
		dequantize_idct_row(in_data + y * width, prediction + y * width, width, height, y,
				out_data + y * width, quantization);
	}
}

void dct_quantize(uint8_t *in_data, uint8_t *prediction, uint32_t width, uint32_t height,
		int16_t *out_data, uint8_t *quantization)
{
	int y;

	for (y = 0; y < height; y += 8)
	{
		dct_quantize_row(in_data + y * width, prediction + y * width, width, height,
				out_data + y * width, quantization);
	}
}
