#ifndef C63_C63_H_
#define C63_C63_H_

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#define MAX_FILELENGTH 200
#define DEFAULT_OUTPUT_FILE "a.mjpg"

#define ISQRT2 0.70710678118654f
#define PI 3.14159265358979
#define ILOG2 1.442695040888963 // 1/log(2);

#define COLOR_COMPONENTS 3

#define Y_COMPONENT 0
#define U_COMPONENT 1
#define V_COMPONENT 2

#define YX 2
#define YY 2
#define UX 1
#define UY 1
#define VX 1
#define VY 1

// Motion estimation search range, pixels in every direction
#define ME_RANGE_Y 16
#define ME_RANGE_U (ME_RANGE_Y/2)
#define ME_RANGE_V (ME_RANGE_Y/2)

#define ME_RANGE(c) (c == Y_COMPONENT ? ME_RANGE_Y : (c == U_COMPONENT ? ME_RANGE_U : ME_RANGE_V))

/* The JPEG file format defines several parts and each part is defined by a
 marker. A file always starts with 0xFF and is then followed by a magic number,
 e.g., like 0xD8 in the SOI marker below. Some markers have a payload, and if
 so, the size of the payload is written before the payload itself. */

#define JPEG_DEF_MARKER 0xFF
#define JPEG_SOI_MARKER 0xD8
#define JPEG_DQT_MARKER 0xDB
#define JPEG_SOF_MARKER 0xC0
#define JPEG_DHT_MARKER 0xC4
#define JPEG_SOS_MARKER 0xDA
#define JPEG_EOI_MARKER 0xD9

#define HUFF_AC_ZERO 16
#define HUFF_AC_SIZE 11

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

struct yuv
{
  uint8_t *Y;
  uint8_t *U;
  uint8_t *V;
};

struct dct
{
	int16_t *base;
	int16_t *Ydct;
	int16_t *Udct;
	int16_t *Vdct;
};

typedef struct yuv yuv_t;
typedef struct dct dct_t;

struct entropy_ctx
{
  FILE *fp;
  unsigned int bit_buffer;
  unsigned int bit_buffer_width;
};

struct macroblock
{
  int use_mv;
  int8_t mv_x, mv_y;
}__attribute__((packed));

struct frame
{
  struct segment_yuv *orig_gpu;			// Original input image
  yuv_t *recons_gpu;		// Reconstructed image
  yuv_t *predicted_gpu;		// Predicted frame from intra-prediction

  yuv_t *orig;
  yuv_t *recons;
  yuv_t *predicted;

  dct_t *residuals;   // Difference between original image and predicted frame
  dct_t *residuals_gpu;

  struct macroblock *mbs[COLOR_COMPONENTS];
  struct macroblock *mbs_gpu[COLOR_COMPONENTS];
  
  int keyframe;
};

struct boundaries
{
	const int* __restrict__ left;
	const int* __restrict__ right;
	const int* __restrict__ top;
	const int* __restrict__ bottom;
};

struct c63_common
{
  int width, height;
  int ypw, yph, upw, uph, vpw, vph;

  int padw[COLOR_COMPONENTS];
  int padh[COLOR_COMPONENTS];

  int mb_cols[COLOR_COMPONENTS];
  int mb_rows[COLOR_COMPONENTS];

  uint8_t qp;                         // Quality parameter

  //int me_search_range;	// This is now defined in c63.h

  uint8_t quanttbl[COLOR_COMPONENTS][64];

  struct frame *refframe;
  struct frame *curframe;

  int framenum;

  int keyframe_interval;
  int frames_since_keyframe;

  struct entropy_ctx e_ctx;

  struct boundaries me_boundaries[COLOR_COMPONENTS];
};

#endif  /* C63_C63_H_ */
