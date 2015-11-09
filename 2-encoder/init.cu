#include "init.h"
#include "init_cuda.h"

static struct boundaries init_me_boundaries(int w, int h, int cols, int rows, int range)
{
	int* lefts = new int[cols];
	int* rights = new int[cols];
	int* tops = new int[rows];
	int* bottoms = new int[rows];

	for (int mb_x = 0; mb_x < cols; ++mb_x)
	{
		lefts[mb_x] = mb_x * 8 - range;
		rights[mb_x] = mb_x * 8 + range;

		if (lefts[mb_x] < 0)
		{
			lefts[mb_x] = 0;
		}

		if (rights[mb_x] > (w - 8))
		{
			rights[mb_x] = w - 8;
		}
	}

	for (int mb_y = 0; mb_y < rows; ++mb_y)
	{
		tops[mb_y] = mb_y * 8 - range;
		bottoms[mb_y] = mb_y * 8 + range;

		if (tops[mb_y] < 0)
		{
			tops[mb_y] = 0;
		}

		if (bottoms[mb_y] > (h - 8))
		{
			bottoms[mb_y] = h - 8;
		}
	}

	struct boundaries boundaries;
	boundaries.left = lefts;
	boundaries.right = rights;
	boundaries.top = tops;
	boundaries.bottom = bottoms;

	return boundaries;
}

static void cleanup_me_boundaries(struct boundaries& boundaries)
{
	delete[] boundaries.left;
	delete[] boundaries.right;
	delete[] boundaries.top;
	delete[] boundaries.bottom;
}

void init_me_boundaries(struct c63_common* cm)
{
	for (int i = 0; i < COLOR_COMPONENTS; ++i)
	{
		int w = cm->padw[i];
		int h = cm->padh[i];
		int cols = cm->mb_cols[i];
		int rows = cm->mb_rows[i];

		cm->me_boundaries[i] = init_me_boundaries(w, h, cols, rows,	ME_RANGE(i));
	}
}

void cleanup_me_boundaries(struct c63_common* cm)
{
	for (int i = 0; i < COLOR_COMPONENTS; ++i)
	{
		cleanup_me_boundaries(cm->me_boundaries[i]);
	}
}
