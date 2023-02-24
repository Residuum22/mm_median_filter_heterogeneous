#include "omp.h"
#include "immintrin.h"
#include "defs.h"


#define swap_min_max_element(x,y) {				\
	__m256i a = _mm256_min_epi8(d[x], d[y]);	\
	__m256i b = _mm256_max_epi8(d[x], d[y]);	\
	d[x] = a; d[y] = b;							\
	}

static inline void compare_swap_5element(__m256i* d) {
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

static inline void compare_swap_4element(__m256i* d) {
	swap_min_max_element(0, 1);
	swap_min_max_element(2, 3);
	swap_min_max_element(0, 2);
	swap_min_max_element(1, 3);
	swap_min_max_element(1, 2);
}

static inline void compare_swap_3element(__m256i* d) {
	swap_min_max_element(1, 2);
	swap_min_max_element(0, 2);
	swap_min_max_element(0, 1);
}

#undef swap_min_max_element


static inline void median_filter_avx_for_component(__m256i* a_matrix)
{

	/* Median value */
	unsigned int M = (FILTER_W - 1) / 2;

	/* Sort colums */
	for (int c = 0; c < FILTER_W; c++)
	{
		__m256i tmp[5] = { a_matrix[c + 0], a_matrix[c + 5], a_matrix[c + 10], a_matrix[c + 15], a_matrix[c + 20] };
		compare_swap_5element(tmp);
		a_matrix[c + 0] = tmp[0];
		a_matrix[c + 5] = tmp[1];
		a_matrix[c + 10] = tmp[2];
		a_matrix[c + 15] = tmp[3];
		a_matrix[c + 20] = tmp[4];
	}

	/* Partially sort rows */
	// Again it is not partial...
	for (int r = 0; r < FILTER_H; r++)
	{
		__m256i tmp[5] = { a_matrix[r * 5 + 0], a_matrix[r * 5 + 1], a_matrix[r * 5 + 2], a_matrix[r * 5 + 3], a_matrix[r * 5 + 4] };
		compare_swap_5element(tmp);
		a_matrix[r * 5 + 0] = tmp[0];
		a_matrix[r * 5 + 1] = tmp[1];
		a_matrix[r * 5 + 2] = tmp[2];
		a_matrix[r * 5 + 3] = tmp[3];
		a_matrix[r * 5 + 4] = tmp[4];
	}

	/* Partially sor diagonals of slope k element of [1:M] */
	/*******************************************************/
	/* k = 1 */
	/*******************************************************/

	/* s element of { 3; 4; 5 } */
	/* 1*r + c = s */
	/* if s = 3 A matrix element are the following in the diagonal (this is located upper main diagonal) */
	__m256i tmp[4] = { a_matrix[3], a_matrix[7], a_matrix[11], a_matrix[15] };
	compare_swap_4element(tmp);
	a_matrix[3] = tmp[0];
	a_matrix[7] = tmp[1];
	a_matrix[11] = tmp[2];
	a_matrix[15] = tmp[3];

	/* if s = 4 A matrix element are the following in the diagonal (this is the main diagonal) */
	__m256i tmp2[5] = { a_matrix[4], a_matrix[8], a_matrix[12], a_matrix[16], a_matrix[20] };
	compare_swap_5element(tmp2);
	a_matrix[4] = tmp2[0];
	a_matrix[8] = tmp2[1];
	a_matrix[12] = tmp2[2];
	a_matrix[16] = tmp2[3];
	a_matrix[20] = tmp2[3];

	/* if s = 5  A matrix element are the following in the diagonal (this is located under main diagonal) */
	__m256i tmp3[4] = { a_matrix[9], a_matrix[13], a_matrix[17], a_matrix[21] };
	compare_swap_4element(tmp3);
	a_matrix[9] = tmp3[0];
	a_matrix[13] = tmp3[1];
	a_matrix[17] = tmp3[2];
	a_matrix[21] = tmp3[3];

	/*******************************************************/
	/* k = 2 */
	/*******************************************************/
	/* 2*r + c = s */
	__m256i tmp4[3] = { a_matrix[9],a_matrix[12],a_matrix[15] };
	compare_swap_3element(tmp);
	a_matrix[9] = tmp4[0];
	a_matrix[12] = tmp4[1];
	a_matrix[15] = tmp4[2];
}

void median_filter_avx(int imgHeight, int imgWidth, int imgHeightF, int imgWidthF, unsigned char* imgSrc, unsigned char* imgDst)
{
#if USE_OMP == 1
#pragma omp parallel for
#endif
	// Image processing
	for (int row = 0; row < imgHeight; row++)
	{	// color componenet in every pixel
		// read has padding so it is wider
		int wr_base = row * imgWidth * 3;
		int rd_base = row * imgWidthF * 3;
		// col is 32 bit this is the elments of the row. 32 because 256 bit avx can handle
		//  32 pieces of 8 bit variable.
		for (int col = 0; col < imgWidth * 3; col += 32, rd_base += 32, wr_base += 32)
		{
			int rd_offset = 0;
			__m256i temp[25];
			// Pixel querying
			for (int fy = 0, index = 0; fy < FILTER_H; fy++)
			{
				int pixel_src = rd_base + rd_offset;
				for (int fx = 0; fx < FILTER_W; fx++, index++)
				{
					// 256 bit read from the picture
					temp[index] = _mm256_loadu_si256((__m256i*) (imgSrc + pixel_src + (fx * 3)));
				}
				// New row for reading and for the 5x5 filter.
				rd_offset += imgWidthF * 3;
			}

			// Filtering
			median_filter_avx_for_component(temp);
			// Output image filling with valid data
			_mm256_stream_si256((__m256i*)(imgDst + wr_base), temp[12]);
			//_mm256_storeu_epi8(&imgDst[wr_base], temp[12]);
		}
	}
}