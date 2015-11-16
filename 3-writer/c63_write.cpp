#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "c63.h"
#include "c63_write.h"
#include "tables.h"

using namespace std;


int frequencies[2][12];
static unsigned int bit_buffer = 0;
static unsigned int bit_buffer_width = 0;

static inline void put_byte(vector<uint8_t>& byte_vector, int byte)
{
	byte_vector.push_back(byte);
}

static inline void put_bytes(vector<uint8_t>& byte_vector, const void* data, unsigned int len)
{
	const uint8_t* bytes = (const uint8_t*) data;
	for (unsigned int i = 0; i < len; ++i)
	{
		byte_vector.push_back(bytes[i]);
	}
}

/**
 * Adds a bit to the bitBuffer. A call to bit_flush() is needed
 * in order to write any remaining bits in the buffer before
 * writing using another function.
 */
static inline void put_bits(vector<uint8_t>& byte_vector, uint16_t bits,
		uint8_t n)
{
	assert(n <= 24 && "Error writing bit");

	if (n == 0)
	{
		return;
	}

	bit_buffer <<= n;
	bit_buffer |= bits & ((1 << n) - 1);
	bit_buffer_width += n;

	while (bit_buffer_width >= 8)
	{
		uint8_t b = (uint8_t) (bit_buffer >> (bit_buffer_width - 8));

		put_byte(byte_vector, b);

		if (b == 0xff)
		{
			put_byte(byte_vector, 0);
		}

		bit_buffer_width -= 8;
	}
}

/**
 * Flushes the bitBuffer by writing zeroes to fill a full byte
 */
static inline void flush_bits(vector<uint8_t>& byte_vector)
{
	if (bit_buffer > 0)
	{
		uint8_t b = bit_buffer << (8 - bit_buffer_width);
		put_byte(byte_vector, b);

		if (b == 0xff)
		{
			put_byte(byte_vector, 0);
		}
	}

	bit_buffer = 0;
	bit_buffer_width = 0;
}

/* Start of Image (SOI) marker, contains no payload. */
static void write_SOI(struct c63_common *cm, vector<uint8_t>& byte_vector)
{
	put_byte(byte_vector, JPEG_DEF_MARKER);
	put_byte(byte_vector, JPEG_SOI_MARKER);
}

/* Define Quatization Tables (DQT) marker, contains the tables as payload. */
static void write_DQT(struct c63_common *cm, vector<uint8_t>& byte_vector)
{
	int16_t size = 2 + (3 * 64 + 1);

	put_byte(byte_vector, JPEG_DEF_MARKER);
	put_byte(byte_vector, JPEG_DQT_MARKER);

	/* Length of segment */
	put_byte(byte_vector, size >> 8);
	put_byte(byte_vector, size & 0xff);

	/* Quatization table for Y component */
	put_byte(byte_vector, Y_COMPONENT);
	put_bytes(byte_vector, cm->quanttbl[Y_COMPONENT], 64);

	/* Quantization table for U component */
	put_byte(byte_vector, U_COMPONENT);
	put_bytes(byte_vector, cm->quanttbl[U_COMPONENT], 64);

	/* Quantization table for V component */
	put_byte(byte_vector, V_COMPONENT);
	put_bytes(byte_vector, cm->quanttbl[V_COMPONENT], 64);
}

/* Start of Frame (SOF) marker with baseline DCT (aka SOF0). */
static void write_SOF0(struct c63_common *cm, vector<uint8_t>& byte_vector)
{
	int16_t size = 8 + 3 * COLOR_COMPONENTS + 1;

	put_byte(byte_vector, JPEG_DEF_MARKER);
	put_byte(byte_vector, JPEG_SOF_MARKER);

	/* Lenght of segment */
	put_byte(byte_vector, size >> 8);
	put_byte(byte_vector, size & 0xff);

	/* Precision */
	put_byte(byte_vector, 8);

	/* Width and height */
	put_byte(byte_vector, cm->height >> 8);
	put_byte(byte_vector, cm->height & 0xff);
	put_byte(byte_vector, cm->width >> 8);
	put_byte(byte_vector, cm->width & 0xff);

	put_byte(byte_vector, COLOR_COMPONENTS);

	put_byte(byte_vector, 1); /* Component id */
	put_byte(byte_vector, 0x22); /* hor | ver sampling factor */
	put_byte(byte_vector, 0); /* Quant. tbl. id */

	put_byte(byte_vector, 2); /* Component id */
	put_byte(byte_vector, 0x11); /* hor | ver sampling factor */
	put_byte(byte_vector, 1); /* Quant. tbl. id */

	put_byte(byte_vector, 3); /* Component id */
	put_byte(byte_vector, 0x11); /* hor | ver sampling factor */
	put_byte(byte_vector, 2); /* Quant. tbl. id */

	/* Is this a keyframe or not? */
	put_byte(byte_vector, cm->curframe->keyframe);
}

static void write_DHT_HTS(struct c63_common *cm, vector<uint8_t>& byte_vector, uint8_t id,
		uint8_t *numlength, uint8_t* data)
{
	/* Find out how many codes we are to write */
	int i, n = 0;

	for (i = 0; i < 16; ++i)
	{
		n += numlength[i];
	}

	put_byte(byte_vector, id);
	put_bytes(byte_vector, numlength, 16);
	put_bytes(byte_vector, data, n);
}

/* Define Huffman Table (DHT) marker, the payload is the Huffman table
 specifiation. */
static void write_DHT(struct c63_common *cm, vector<uint8_t>& byte_vector)
{
	int16_t size = 0x01A2; /* 2 + n*(17+mi); */

	put_byte(byte_vector, JPEG_DEF_MARKER);
	put_byte(byte_vector, JPEG_DHT_MARKER);

	/* Length of segment */
	put_byte(byte_vector, size >> 8);
	put_byte(byte_vector, size & 0xff);

	/* Write the four huffman table specifications */
	/* DC table 0 */
	write_DHT_HTS(cm, byte_vector, 0x00, DCVLC_num_by_length[0], DCVLC_data[0]);
	/* DC table 1 */
	write_DHT_HTS(cm, byte_vector, 0x01, DCVLC_num_by_length[1], DCVLC_data[1]);
	/* AC table 0 */
	write_DHT_HTS(cm, byte_vector, 0x10, ACVLC_num_by_length[0], ACVLC_data[0]);
	/* AC table 1 */
	write_DHT_HTS(cm, byte_vector, 0x11, ACVLC_num_by_length[1], ACVLC_data[1]);
}

/* Start of Scan (SOS) marker, the payload is references to the huffman
 tables. It is followed by the image data, see write_frame(). */
static void write_SOS(struct c63_common *cm, vector<uint8_t>& byte_vector)
{
	int16_t size = 6 + 2 * COLOR_COMPONENTS;

	put_byte(byte_vector, JPEG_DEF_MARKER);
	put_byte(byte_vector, JPEG_SOS_MARKER);

	/* Length of the segment */
	put_byte(byte_vector, size >> 8);
	put_byte(byte_vector, size & 0xff);

	put_byte(byte_vector, COLOR_COMPONENTS);

	put_byte(byte_vector, 1); /* Component id */
	put_byte(byte_vector, 0x00); /* DC | AC huff tbl */
	put_byte(byte_vector, 2); /* Component id */
	put_byte(byte_vector, 0x11); /* DC | AC huff tbl */
	put_byte(byte_vector, 3); /* Component id */
	put_byte(byte_vector, 0x11); /* DC | AC huff tbl */

	put_byte(byte_vector, 0); /* ss, first AC */
	put_byte(byte_vector, 63); /* se, last AC */
	put_byte(byte_vector, 0); /* ah | al */
}

/* End of Image (EOI) marker, contains no payload. */
static void write_EOI(struct c63_common *cm, vector<uint8_t>& byte_vector)
{
	put_byte(byte_vector, JPEG_DEF_MARKER);
	put_byte(byte_vector, JPEG_EOI_MARKER);
}

static inline uint8_t bit_width(int16_t i)
{
	if (__builtin_expect(!i, 0))
	{
		return 0;
	}

	return 32 - __builtin_clz(abs(i));
}

static void write_block(struct c63_common *cm, vector<uint8_t>& byte_vector, int16_t *in_data,
		uint32_t width, uint32_t height, uint32_t uoffset, uint32_t voffset, int16_t *prev_DC,
		int32_t cc, int channel)
{
	uint32_t i, j;

	/* Write motion vector */
	struct macroblock *mb = &cm->curframe->mbs[channel][voffset / 8 * cm->padw[channel] / 8
			+ uoffset / 8];

	/* Use inter pred? */
	put_bits(byte_vector, mb->use_mv, 1);

	if (mb->use_mv)
	{
		int reuse_prev_mv = 0;

		if (uoffset && (mb - 1)->use_mv && (mb - 1)->mv_x == mb->mv_x && (mb - 1)->mv_y == mb->mv_y)
		{
			reuse_prev_mv = 1;
		}

		put_bits(byte_vector, reuse_prev_mv, 1);

		if (!reuse_prev_mv)
		{
			uint8_t sz;
			int16_t val;

			/* Encode MV x-coord */
			val = mb->mv_x;
			sz = bit_width(val);
			if (val < 0)
			{
				--val;
			}

			put_bits(byte_vector, MVVLC[sz], MVVLC_Size[sz]);
			put_bits(byte_vector, val, sz);
			/* ++frequencies[cc][sz]; */

			/* Encode MV y-coord */
			val = mb->mv_y;
			sz = bit_width(val);
			if (val < 0)
			{
				--val;
			}

			put_bits(byte_vector, MVVLC[sz], MVVLC_Size[sz]);
			put_bits(byte_vector, val, sz);
			/* ++frequencies[cc][sz]; */
		}
	}

	/* Write residuals */

	/* Residuals stored linear in memory */
	int16_t *block = &in_data[uoffset * 8 + voffset * width];
	int32_t num_ac = 0;

#if 0
	static int blocknum;
	++blocknum;

	printf("Dump block %d:\n", blocknum);

	for(i=0; i<8; ++i)
	{
		for (j=0; j<8; ++j)
		{
			printf(", %5d", block[i*8+j]);
		}
		printf("\n");
	}

	printf("Finished block\n\n");
#endif

	/* Calculate DC component, and write to stream */
	int16_t dc = block[0] - *prev_DC;
	*prev_DC = block[0];

	uint8_t size = bit_width(dc);
	put_bits(byte_vector, DCVLC[cc][size], DCVLC_Size[cc][size]);

	if (dc < 0)
	{
		dc = dc - 1;
	}
	put_bits(byte_vector, dc, size);

	/* find the last nonzero entry of the ac-coefficients */
	for (j = 64; j > 1 && !block[j - 1]; j--)
		;

	/* Put the nonzero ac-coefficients */
	for (i = 1; i < j; i++)
	{
		int16_t ac = block[i];
		if (ac == 0)
		{
			if (++num_ac == 16)
			{
				put_bits(byte_vector, ACVLC[cc][15][0], ACVLC_Size[cc][15][0]);
				num_ac = 0;
			}
		}
		else
		{
			uint8_t size = bit_width(ac);
			put_bits(byte_vector, ACVLC[cc][num_ac][size],
					ACVLC_Size[cc][num_ac][size]);

			if (ac < 0)
			{
				--ac;
			}

			put_bits(byte_vector, ac, size);
			num_ac = 0;
		}
	}

	/* Put end of block marker */
	if (j < 64)
	{
		put_bits(byte_vector, ACVLC[cc][0][0], ACVLC_Size[cc][0][0]);
	}
}

static void write_interleaved_data_MCU(struct c63_common *cm, vector<uint8_t>& byte_vector,
		int16_t *dct, uint32_t wi, uint32_t he, uint32_t h, uint32_t v, uint32_t x, uint32_t y,
		int16_t *prev_DC, int32_t cc, int channel)
{
	uint32_t i, j, ii, jj;

	for (j = y * v * 8; j < (y + 1) * v * 8; j += 8)
	{
		jj = he - 8;
		jj = MIN(j, jj);

		for (i = x * h * 8; i < (x + 1) * h * 8; i += 8)
		{
			ii = wi - 8;
			ii = MIN(i, ii);

			write_block(cm, byte_vector, dct, wi, he, ii, jj, prev_DC, cc, channel);
		}
	}
}

static void write_interleaved_data(struct c63_common *cm, vector<uint8_t>& byte_vector)
{
	int16_t prev_DC[3] = { 0, 0, 0 };
	uint32_t u, v;

	/* Set up which huffman tables we want to use */
	int32_t yhtbl = 0;
	int32_t uhtbl = 1;
	int32_t vhtbl = 1;

	/* Find the number of MCU's for the intensity */
	uint32_t ublocks = (uint32_t) (ceil(cm->ypw / (float) (8.0f * YX)));
	uint32_t vblocks = (uint32_t) (ceil(cm->yph / (float) (8.0f * YY)));

	/* Write the MCU's interleaved */
	for (v = 0; v < vblocks; ++v)
	{
		for (u = 0; u < ublocks; ++u)
		{
			write_interleaved_data_MCU(cm, byte_vector, cm->curframe->residuals->Ydct, cm->ypw,
					cm->yph, YX, YY, u, v, &prev_DC[0], yhtbl, 0);
			write_interleaved_data_MCU(cm, byte_vector, cm->curframe->residuals->Udct, cm->upw,
					cm->uph, UX, UY, u, v, &prev_DC[1], uhtbl, 1);
			write_interleaved_data_MCU(cm, byte_vector, cm->curframe->residuals->Vdct, cm->vpw,
					cm->vph, VX, VY, u, v, &prev_DC[2], vhtbl, 2);
		}
	}

	flush_bits(byte_vector);
}

vector<uint8_t> write_frame_to_buffer(struct c63_common *cm)
{
	vector<uint8_t> byte_vector;

	/* Write headers */

	/* Start Of Image */
	write_SOI(cm, byte_vector);
	/* Define Quantization Table(s) */
	write_DQT(cm, byte_vector);
	/* Start Of Frame 0(Baseline DCT) */
	write_SOF0(cm, byte_vector);
	/* Define Huffman Tables(s) */
	write_DHT(cm, byte_vector);
	/* Start of Scan */
	write_SOS(cm, byte_vector);

	write_interleaved_data(cm, byte_vector);

	/* End Of Image */
	write_EOI(cm, byte_vector);

	return byte_vector;
}

void write_buffer_to_file(const vector<uint8_t>& byte_vector, FILE* file)
{
	size_t n = fwrite(&byte_vector[0], sizeof(uint8_t), byte_vector.size(), file);

	if (n != byte_vector.size())
	{
		fprintf(stderr, "Error writing bytes\n");
		exit(EXIT_FAILURE);
	}
}
