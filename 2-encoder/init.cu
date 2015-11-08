#include "init.h"
#include "init_cuda.h"

static struct boundaries init_me_boundaries(int w, int h, int cols, int rows, int range)
{
	int* lefts = new int[cols];
	int* rights = new int[cols];
	int* tops = new int[rows];
	int* bottoms = new int[rows];

	for (int mb_x = 0; mb_x < cols; ++mb_x) {
		lefts[mb_x] = mb_x * 8 - range;
		rights[mb_x] = mb_x * 8 + range;

		if (lefts[mb_x] < 0) {
			lefts[mb_x] = 0;
		}

		if (rights[mb_x] > (w - 8)) {
			rights[mb_x] = w - 8;
		}
	}

	for (int mb_y = 0; mb_y < rows; ++mb_y) {
		tops[mb_y] = mb_y * 8 - range;
		bottoms[mb_y] = mb_y * 8 + range;

		if (tops[mb_y] < 0) {
			tops[mb_y] = 0;
		}

		if (bottoms[mb_y] > (h - 8)) {
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

static void cleanup_me_boundaries(struct boundaries* boundaries)
{
	delete[] boundaries->left;
	delete[] boundaries->right;
	delete[] boundaries->top;
	delete[] boundaries->bottom;
}

void init_boundaries(struct c63_common* cm)
{
	struct boundaries boundaries[3];
	int Y = Y_COMPONENT;
	int U = U_COMPONENT;
	int V = V_COMPONENT;

	boundaries[Y] = init_me_boundaries(cm->ypw, cm->yph, cm->mb_colsY, cm->mb_rowsY, ME_RANGE_Y);
	boundaries[U] = init_me_boundaries(cm->upw, cm->uph, cm->mb_colsU, cm->mb_rowsU, ME_RANGE_U);
	boundaries[V] = init_me_boundaries(cm->vpw, cm->vph, cm->mb_colsV, cm->mb_rowsV, ME_RANGE_V);

	cm->me_boundariesY = init_me_boundaries_gpu(&boundaries[Y], cm->mb_colsY, cm->mb_rowsY, cm->cuda_data.streamY);
	cm->me_boundariesU = init_me_boundaries_gpu(&boundaries[U], cm->mb_colsU, cm->mb_rowsU, cm->cuda_data.streamU);
	cm->me_boundariesV = init_me_boundaries_gpu(&boundaries[V], cm->mb_colsV, cm->mb_rowsV, cm->cuda_data.streamV);

	cleanup_me_boundaries(&boundaries[Y]);
	cleanup_me_boundaries(&boundaries[U]);
	cleanup_me_boundaries(&boundaries[V]);
}

void cleanup_boundaries(struct c63_common* cm)
{
	cleanup_me_boundaries_gpu(&cm->me_boundariesY);
	cleanup_me_boundaries_gpu(&cm->me_boundariesU);
	cleanup_me_boundaries_gpu(&cm->me_boundariesV);
}
