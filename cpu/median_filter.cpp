#include "omp.h"
#include "emmintrin.h"
#include "nmmintrin.h"
#include "immintrin.h"
#include "defs.h"

#define min(x, y) (x < y ? x : y) 
#define max(x, y) (x < y ? y : x) 
// Sort x and y element of the matrix in increment order.
// The naming is unlucky but I refered to the min max macros.
#define swap_min_max_element(x,y) {	\
	const int a = min(d[x], d[y]);	\
	const int b = max(d[x], d[y]);	\
	d[x] = a; d[y] = b;				\
	}

// The IEEE compare swap algorith for 5, 4, and 3 element
static inline void compare_swap_5element(int* d) {
	swap_min_max_element(0, 1);
	swap_min_max_element(3, 4);
	swap_min_max_element(2, 4);
	swap_min_max_element(2, 3);
	swap_min_max_element(0, 3);
	swap_min_max_element(0, 2);
	swap_min_max_element(1, 4);
	swap_min_max_element(1, 3);
	swap_min_max_element(1, 2);
}

static inline void compare_swap_4element(int* d) {
	swap_min_max_element(0, 1);
	swap_min_max_element(2, 3);
	swap_min_max_element(0, 2);
	swap_min_max_element(1, 3);
	swap_min_max_element(1, 2);
}

static inline void compare_swap_3element(int* d) {
	swap_min_max_element(1, 2);
	swap_min_max_element(0, 2);
	swap_min_max_element(0, 1);
}

static void median_filter_for_component(unsigned char* a_matrix)
{
	/* Median value */
	unsigned int M = (FILTER_W - 1) / 2;

	/* Sort colums */
	for (int current_colum = 0; current_colum < FILTER_W; current_colum++)
	{
		int tmp[5] = { a_matrix[current_colum + 0], a_matrix[current_colum + 5], a_matrix[current_colum + 10], a_matrix[current_colum + 15], a_matrix[current_colum + 20] };
		compare_swap_5element(tmp);
		a_matrix[current_colum + 0] = tmp[0];
		a_matrix[current_colum + 5] = tmp[1];
		a_matrix[current_colum + 10] = tmp[2];
		a_matrix[current_colum + 15] = tmp[3];
		a_matrix[current_colum + 20] = tmp[4];
	}

	/* Partially sort rows */
	// Not partial just the figure 2.3 call this part partially sort rows. 
	for (int current_row = 0; current_row < FILTER_H; current_row++)
	{
		int tmp[5] = { a_matrix[current_row * 5 + 0], a_matrix[current_row * 5 + 1], a_matrix[current_row * 5 + 2], a_matrix[current_row * 5 + 3], a_matrix[current_row * 5 + 4] };
		compare_swap_5element(tmp);
		a_matrix[current_row * 5 + 0] = tmp[0];
		a_matrix[current_row * 5 + 1] = tmp[1];
		a_matrix[current_row * 5 + 2] = tmp[2];
		a_matrix[current_row * 5 + 3] = tmp[3];
		a_matrix[current_row * 5 + 4] = tmp[4];
	}

	/* Partially sor diagonals of slope k element of [1:M] */
	/*******************************************************/
	/* k = 1 */
	/*******************************************************/

	/* s element of { 3; 4; 5 } */
	/* 1*r + c = s */
	/* if s = 3 A matrix element are the following in the diagonal (this is located upper main diagonal) */
	int tmp[4] = { a_matrix[3], a_matrix[7], a_matrix[11], a_matrix[15] };
	compare_swap_4element(tmp);
	a_matrix[3]  = tmp[0];
	a_matrix[7]  = tmp[1];
	a_matrix[11] = tmp[2];
	a_matrix[15] = tmp[3];

	/* if s = 4 A matrix element are the following in the diagonal (this is the main diagonal) */
	int tmp2[5] = { a_matrix[4], a_matrix[8], a_matrix[12], a_matrix[16], a_matrix[20] };
	compare_swap_5element(tmp2);
	a_matrix[4]  = tmp2[0];
	a_matrix[8]  = tmp2[1];
	a_matrix[12] = tmp2[2];
	a_matrix[16] = tmp2[3];
	a_matrix[20] = tmp2[3];

	/* if s = 5  A matrix element are the following in the diagonal (this is located under main diagonal) */
	int tmp3[4] = { a_matrix[9], a_matrix[13], a_matrix[17], a_matrix[21] };
	compare_swap_4element(tmp3);
	a_matrix[9]  = tmp3[0];
	a_matrix[13] = tmp3[1];
	a_matrix[17] = tmp3[2];
	a_matrix[21] = tmp3[3];

	/*******************************************************/
	/* k = 2 */
	/*******************************************************/
	/* 2*r + c = s */
	int tmp4[3] = { a_matrix[9],a_matrix[12],a_matrix[15] };
	compare_swap_3element(tmp);
	a_matrix[9]  = tmp4[0];
	a_matrix[12] = tmp4[1];
	a_matrix[15] = tmp4[2];
}

void median_filter(int imgHeight, int imgWidth, int imgHeightF, int imgWidthF, unsigned char* imgSrc, unsigned char* imgDst)
{
#if USE_OMP == 1
#pragma omp parallel for
#endif
	// Image processing part this is like on labors.
	for (int row = 0; row < imgHeight; row++)
	{
		int wr_base = row * imgWidth * 3;
		int rd_base = row * imgWidthF * 3;

		for (int col = 0; col < imgWidth; col++)
		{
			// Median filter 5x5 matrix in 1d -> 25 element
			unsigned char r[25];
			unsigned char g[25];
			unsigned char b[25];

			int rd_offset = 0;
			int k = 0;

			// Pixel querying
			for (int fy = 0; fy < FILTER_H; fy++)
			{
				for (int fx = 0; fx < FILTER_W; fx++)
				{
					// Separate rgb componenet to prepare for filtering.
					int pixel_src = rd_base + rd_offset;
					r[k] = (unsigned char)(*(imgSrc + pixel_src + 0));
					g[k] = (unsigned char)(*(imgSrc + pixel_src + 1));
					b[k] = (unsigned char)(*(imgSrc + pixel_src + 2));
					rd_offset = rd_offset + 3;
					k++;
				}
				rd_offset = rd_offset - 5 * 3 + imgWidthF * 3;
			}
			// Filtering componenets
			median_filter_for_component(r);
			median_filter_for_component(g);
			median_filter_for_component(b);

			// Output image filling with valid data
			// The index is 12 because the indexing begin from 0 and 2x5+2 is the median value.
			*(imgDst + wr_base + 0) = (unsigned char)(r[12]);
			*(imgDst + wr_base + 1) = (unsigned char)(g[12]);
			*(imgDst + wr_base + 2) = (unsigned char)(b[12]);

			// New base for the new pixels in the 
			wr_base = wr_base + 3;
			rd_base = rd_base + 3;
		}
	}
}
