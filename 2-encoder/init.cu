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

void init_boundaries(struct c63_common* cm, const struct c63_cuda& c63_cuda)
{
	static const int Y = Y_COMPONENT;
	static const int U = U_COMPONENT;
	static const int V = V_COMPONENT;

	struct boundaries boundaries[COLOR_COMPONENTS];

	boundaries[Y] = init_me_boundaries(cm->ypw, cm->yph, cm->mb_cols[Y], cm->mb_rows[Y], ME_RANGE_Y);
	boundaries[U] = init_me_boundaries(cm->upw, cm->uph, cm->mb_cols[U], cm->mb_rows[U], ME_RANGE_U);
	boundaries[V] = init_me_boundaries(cm->vpw, cm->vph, cm->mb_cols[V], cm->mb_rows[V], ME_RANGE_V);

	cm->me_boundariesY = init_me_boundaries_gpu(boundaries[Y], cm->mb_cols[Y], cm->mb_rows[Y], c63_cuda.streamY);
	cm->me_boundariesU = init_me_boundaries_gpu(boundaries[U], cm->mb_cols[U], cm->mb_rows[U], c63_cuda.streamU);
	cm->me_boundariesV = init_me_boundaries_gpu(boundaries[V], cm->mb_cols[V], cm->mb_rows[V], c63_cuda.streamV);

	cleanup_me_boundaries(&boundaries[Y]);
	cleanup_me_boundaries(&boundaries[U]);
	cleanup_me_boundaries(&boundaries[V]);
}

void cleanup_boundaries(struct c63_common* cm)
{
	cleanup_me_boundaries_gpu(cm->me_boundariesY);
	cleanup_me_boundaries_gpu(cm->me_boundariesU);
	cleanup_me_boundaries_gpu(cm->me_boundariesV);
}
