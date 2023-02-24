#define FILTER_W 5
#define FILTER_H 5

#define swap_min_max_element(x, y) {\
	const float a = min(d[x], d[y]);        \
	const float b = max(d[x], d[y]);	    \
    d[x] = a; d[y] = b;                     \
	}                                       \

void compare_swap_5element(float* d) {
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

void compare_swap_4element(float* d) {
	swap_min_max_element(0, 1);
	swap_min_max_element(2, 3);
	swap_min_max_element(0, 2);
	swap_min_max_element(1, 3);
	swap_min_max_element(1, 2);
}

void compare_swap_3element(float* d) {
	swap_min_max_element(1, 2);
	swap_min_max_element(0, 2);
	swap_min_max_element(0, 1);
}

__kernel void kernel_median_filter(__global unsigned char* gInput,
                                 __global unsigned char* gOutput,
                                 __constant int *filter_coeffs2,
								 int imgWidth,
								 int imgWidthF)
{
    // sor és oszlop indexek meghatározása 
    int row = get_group_id(1) * get_local_size(1) + get_local_id(1);
    int col = get_group_id(0) * get_local_size(0) + get_local_id(0);

    int out_pix_base = (row * imgWidth + col) * 3; // meghatarozzuk a kimeneti pixelunket

    __local float in_shmem[20][20][3];

    int picture_1D_address = get_local_id(1) * get_local_size(0) + get_local_id(0); 

    int rbga_address = picture_1D_address % 3;
    int col_address = (picture_1D_address / 3) % 20; 
    int row_address = picture_1D_address / 60;

    int picture_1D_address_base = (get_group_id(1) * get_local_size(1)) * 3 * imgWidthF + (get_group_id(0) * get_local_size(0)) * 3 + (row_address * 3 * imgWidthF);
    
    if (picture_1D_address < 3 * 20 * 4)
    {
        #pragma unroll
        for (int ld = 0; ld < 5; ld++)
        {
            in_shmem[row_address + ld * 4][col_address][rbga_address] = (float)(gInput[picture_1D_address_base + (picture_1D_address % 60)]);
            picture_1D_address_base = picture_1D_address_base + imgWidthF * 3 * 4;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float filter_coeffs[25];
    #pragma unroll 3
    for (int rgba = 0; rgba < 3; rgba++)
    {
        #pragma unroll 5
        for (int fy = 0; fy < 5; fy++)
        {
    
            #pragma unroll 5
            for (int fx = 0; fx < 5; fx++)
                {
                    filter_coeffs[5 * fy + fx] = (float)in_shmem[get_local_id(1) + fy][get_local_id(0) + fx][rgba];
                }
        }

        unsigned int M = (FILTER_W - 1) / 2;

        //column
        #pragma unroll 5
        for (int c = 0; c < (FILTER_W); c++)
        {
            float tmp[5] = { filter_coeffs[c],filter_coeffs[c + 5],filter_coeffs[c + 
            10],filter_coeffs[c + 15],filter_coeffs[c + 20] };
            compare_swap_5element(tmp);
            filter_coeffs[c] = tmp[0];
            filter_coeffs[c + 5] = tmp[1];
            filter_coeffs[c + 10] = tmp[2];
            filter_coeffs[c + 15] = tmp[3];
            filter_coeffs[c + 20] = tmp[4];
        }

        //row
        #pragma unroll 5
        for (int r = 0; r < (FILTER_H); r++)
        {
            float tmp[5] = { filter_coeffs[r * 5],filter_coeffs[r * 5 + 1],filter_coeffs[r 
            * 5 + 2],filter_coeffs[r * 5 + 3],filter_coeffs[r * 5 + 4] };
            compare_swap_5element(tmp);
            filter_coeffs[r * 5] = tmp[0];
            filter_coeffs[r * 5 + 1] = tmp[1];
            filter_coeffs[r * 5 + 2] = tmp[2];
            filter_coeffs[r * 5 + 3] = tmp[3];
            filter_coeffs[r * 5 + 4] = tmp[4];
        }

        //diagonals
        float tmp[4] = { filter_coeffs[3],filter_coeffs[7],filter_coeffs[11],filter_coeffs[15] };
        compare_swap_4element(tmp);
        filter_coeffs[3] = tmp[0];
        filter_coeffs[7] = tmp[1];
        filter_coeffs[11] = tmp[2];
        filter_coeffs[15] = tmp[3];


        float tmp2[5] = { filter_coeffs[4],filter_coeffs[8],filter_coeffs[12],filter_coeffs[16],filter_coeffs[20] };
        compare_swap_5element(tmp2);
        filter_coeffs[4] = tmp2[0];
        filter_coeffs[8] = tmp2[1];
        filter_coeffs[12] = tmp2[2];
        filter_coeffs[16] = tmp2[3];
        filter_coeffs[20] = tmp[3];

        float tmp3[4] = { filter_coeffs[9],filter_coeffs[13],filter_coeffs[17],filter_coeffs[21] };
        compare_swap_4element(tmp3);
        filter_coeffs[9] = tmp3[0];
        filter_coeffs[13] = tmp3[1];
        filter_coeffs[17] = tmp3[2];
        filter_coeffs[21] = tmp3[3];

        float tmp4[3] = { filter_coeffs[9],filter_coeffs[12],filter_coeffs[15] };
        compare_swap_3element(tmp4);
        filter_coeffs[9] = tmp4[0];
        filter_coeffs[12] = tmp4[1];
        filter_coeffs[15] = tmp4[2];

        gOutput[out_pix_base + rgba] = (unsigned char)(filter_coeffs[12]);
    }
}