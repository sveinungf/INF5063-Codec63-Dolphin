#include "init.h"
#include "tables.h"
#include "cuda/common.h"


static const int Y = Y_COMPONENT;
static const int U = U_COMPONENT;
static const int V = V_COMPONENT;

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

static void init_me_boundaries(struct c63_common* cm)
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

static void cleanup_me_boundaries(struct c63_common* cm)
{
	for (int i = 0; i < COLOR_COMPONENTS; ++i)
	{
		cleanup_me_boundaries(cm->me_boundaries[i]);
	}
}

struct c63_common* init_c63_common(int width, int height, const struct c63_cuda& c63_cuda)
{
	/* calloc() sets allocated memory to zero */
	struct c63_common *cm = (struct c63_common*) calloc(1, sizeof(struct c63_common));

	cm->width = width;
	cm->height = height;

	cm->padw[Y] = cm->ypw = (uint32_t) (ceil(width / 16.0f) * 16);
	cm->padh[Y] = cm->yph = (uint32_t) (ceil(height / 16.0f) * 16);
	cm->padw[U] = cm->upw = (uint32_t) (ceil(width * UX / (YX * 8.0f)) * 8);
	cm->padh[U] = cm->uph = (uint32_t) (ceil(height * UY / (YY * 8.0f)) * 8);
	cm->padw[V] = cm->vpw = (uint32_t) (ceil(width * VX / (YX * 8.0f)) * 8);
	cm->padh[V] = cm->vph = (uint32_t) (ceil(height * VY / (YY * 8.0f)) * 8);

	cm->mb_cols[Y] = cm->ypw / 8;
	cm->mb_cols[U] = cm->mb_cols[Y] / 2;
	cm->mb_cols[V] = cm->mb_cols[U];

	cm->mb_rows[Y] = cm->yph / 8;
	cm->mb_rows[U] = cm->mb_rows[Y] / 2;
	cm->mb_rows[V] = cm->mb_rows[U];

	/* Quality parameters -- Home exam deliveries should have original values,
	 i.e., quantization factor should be 25, search range should be 16, and the
	 keyframe interval should be 100. */
	cm->qp = 25;                  // Constant quantization factor. Range: [1..50]
	//cm->me_search_range = 16;   // This is now defined in c63.h
	cm->keyframe_interval = 100;  // Distance between keyframes

	/* Initialize quantization tables */
	for (int i = 0; i < 64; ++i)
	{
		cm->quanttbl[Y][i] = yquanttbl_def[i] / (cm->qp / 10.0);
		cm->quanttbl[U][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
		cm->quanttbl[V][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
	}

	init_me_boundaries(cm);

	cm->curframe = create_frame(cm, c63_cuda);
	cm->refframe = create_frame(cm, c63_cuda);

	return cm;
}

void cleanup_c63_common(struct c63_common* cm)
{
	cleanup_me_boundaries(cm);

	destroy_frame(cm->curframe);
	destroy_frame(cm->refframe);

	free(cm);
}
