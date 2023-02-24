
#include "types.h"

#define ImgWidthMax 1600
// Filter változók

#define FILTER_W 5
#define FILTER_H 5

#define min(x, y) (x < y ? x : y)
#define max(x, y) (x < y ? y : x)
#define SWAP(x , y) { 				\
	const int a = min(d[x], d[y]);  \
	const int b = max(d[x], d[y]);  \
	 d[x] = a; d[y] = b;			\
}

static inline void sort5(uint8_t * d) {
	SWAP(0, 4);
	SWAP(0, 2);
	SWAP(1, 3);
	SWAP(2, 4);
	SWAP(0, 1);
	SWAP(2, 3);
	SWAP(1, 4);
	SWAP(1, 2);
	SWAP(3, 4);
}

static inline void sort4(uint8_t * d) {
	SWAP(0, 1);
	SWAP(2, 3);
	SWAP(0, 2);
	SWAP(1, 3);
	SWAP(1, 2);
}

static inline void sort3(uint8_t * d) {
	SWAP(1, 2);
	SWAP(0, 2);
	SWAP(0, 1);
}


uint8_t median_filter(uint8_t *in)
{
	// a szuro minden orajelben kaphat uj bemenetet
	#pragma HLS PIPELINE II=1
	// bemenetrol lokalais masolat
	uint8_t a[FILTER_W*FILTER_H];

	for(int i = 0; i < FILTER_W * FILTER_H; i++)
	{
		#pragma HLS unroll factor=25
		a[i] = in[i];
	}

	//column
	for (int c = 0; c < (FILTER_W); c++)
	{
		#pragma HLS unroll factor=5
		uint8_t tmp[5] = { a[c],a[c+5],a[c + 10],a[c + 15],a[c + 20]};
		sort5(tmp);
		a[c] = tmp[0];
		a[c+5] = tmp[1];
		a[c+10] = tmp[2];
		a[c+15] = tmp[3];
		a[c+20] = tmp[4];
	}

	//row
	for (int r = 0; r < (FILTER_H); r++)
	{
		#pragma HLS unroll factor=5
		uint8_t tmp[5] = { a[r*5],a[r*5 + 1],a[r*5 + 2],a[r*5 + 3],a[r*5 + 4] };
		sort5(tmp);
		a[r * 5] = tmp[0];
		a[r * 5+1] = tmp[1];
		a[r * 5+2] = tmp[2];
		a[r * 5+3] = tmp[3];
		a[r * 5+4] = tmp[4];
	}

	//diagonals
	uint8_t tmp[4] = { a[3],a[7],a[11],a[15]};
	sort4(tmp);
	a[3] = tmp[0];
	a[7] = tmp[1];
	a[11] = tmp[2];
	a[15] = tmp[3];

	uint8_t tmp2[5] = { a[4],a[8],a[12],a[16],a[20] };
	sort5(tmp2);
	a[4] = tmp2[0];
	a[8] = tmp2[1];
	a[12] = tmp2[2];
	a[16] = tmp2[3];
	a[20] = tmp[3];

	uint8_t tmp3[4] = { a[9],a[13],a[17],a[21] };
	sort4(tmp3);
	a[9] = tmp3[0];
	a[13] = tmp3[1];
	a[17] = tmp3[2];
	a[21] = tmp3[3];

	uint8_t tmp4[3] = { a[9],a[12],a[15]};
	sort3(tmp4);
	a[9] = tmp4[0];
	a[12] = tmp4[1];
	a[15] = tmp4[2];

	return a[12];
}

void median_top(uint8_t *r_in, uint8_t *g_in, uint8_t *b_in, bool *hs_in, bool *vs_in, bool *de_in,
 uint8_t *r_out, uint8_t *g_out, uint8_t *b_out, bool *hs_out, bool *vs_out, bool *de_out)
{
	#pragma HLS INTERFACE ap_none port=r_in
	#pragma HLS INTERFACE ap_none port=g_in
	#pragma HLS INTERFACE ap_none port=b_in
	#pragma HLS INTERFACE ap_none port=hs_in
	#pragma HLS INTERFACE ap_none port=vs_in
	#pragma HLS INTERFACE ap_none port=de_in
	#pragma HLS INTERFACE ap_none port=r_out
	#pragma HLS INTERFACE ap_none port=g_out
	#pragma HLS INTERFACE ap_none port=b_out
	#pragma HLS INTERFACE ap_none port=hs_out
	#pragma HLS INTERFACE ap_none port=vs_out
	#pragma HLS INTERFACE ap_none port=de_out
	#pragma HLS INTERFACE ap_ctrl_none port=return

	//100MHz-es pixelfrekvencia és 100MHz-es órajelfrekvencia mellett a real-time működésmegköveteli, hogy minden órajelben tudjunk
	//új adatot fogadni a bemeneten
	#pragma HLS PIPELINE II=1

	// a szűrő kernel kapja mindig a frissen beérkezett pixel értéket, valamint buffereljük az előző négy sort
	//ami információ a szűréshez kell. Minden új pixel érkezésekor a kernelt eggyel jobbra toljuk, hogy kijöjjön az 5x5-ös pixelhalmaz
	//amit aztán szűrünk. Mivel a későbiekben szükség van még egyéb pixelek számításánál is aértékekre, ezért a rögtön a kernelbe másolt
	//pixelt átmásoljuk a BRAM bufferekbe is.
	static uint8_t red_row_0[ImgWidthMax] ={0};
	static uint8_t red_row_1[ImgWidthMax] ={0};
	static uint8_t red_row_2[ImgWidthMax] ={0};
	static uint8_t red_row_3[ImgWidthMax] ={0};

	static uint8_t green_row_0[ImgWidthMax] ={0};
	static uint8_t green_row_1[ImgWidthMax] ={0};
	static uint8_t green_row_2[ImgWidthMax] ={0};
	static uint8_t green_row_3[ImgWidthMax] ={0};

	static uint8_t blue_row_0[ImgWidthMax] ={0};
	static uint8_t blue_row_1[ImgWidthMax] ={0};
	static uint8_t blue_row_2[ImgWidthMax] ={0};
	static uint8_t blue_row_3[ImgWidthMax] ={0};

	// A 3 színkomponensnek 3 db 25 méertű tároló(DFF)
	static uint8_t red_kernel[FILTER_W * FILTER_H] ={0};
	#pragma HLS ARRAY_PARTITION variable=red_kernel complete dim=0
	static uint8_t green_kernel[FILTER_W * FILTER_H] ={0};
	#pragma HLS ARRAY_PARTITION variable=green_kernel complete dim=0
	static uint8_t blue_kernel[FILTER_W * FILTER_H] ={0};
	#pragma HLS ARRAY_PARTITION variable=blue_kernel complete dim=0
	//A későbbi szűrőablakok számára kimentjük az eredeti pixelértékeket is feldolgozásra
	uint8_t red_kernel_tmp[FILTER_W * FILTER_H];
	#pragma HLS ARRAY_PARTITION variable=red_kernel_tmp complete dim=0
	uint8_t green_kernel_tmp[FILTER_W * FILTER_H];
	#pragma HLS ARRAY_PARTITION variable=green_kernel_tmp complete dim=0
	uint8_t blue_kernel_tmp[FILTER_W * FILTER_H];
	#pragma HLS ARRAY_PARTITION variable=blue_kernel_tmp complete dim=0
	// pixel indek nyilvátartás, és a hs jel felfutóéldetektora

	static int x_index;
	static int y_index;
	static bool hs_in_prev = 0;
	// Egy új kép érkezésekor nullázni kell az indexeket
	if(*vs_in == 1)
	{
		x_index = 0;
		y_index = 0;
	}

	// Egy új sor érkezésekor nullázni kell az x indexet és növelni az y indexet
	if(*hs_in == 1 && hs_in_prev == 0)
	{
		x_index = 0;
		y_index++;
	}

	// Ha egy új pixel érkezett
	if(*de_in == 1)
	{
		red_kernel[0] = *r_in;
		green_kernel[0] = *g_in;
		blue_kernel[0] = *b_in;

		// A jobb szélső szűrő oszlopot beolvassuk a sorbuffer megfelelő indexű helyéről
		red_kernel[20] = red_row_0[x_index];
		green_kernel[20] = green_row_0[x_index];
		blue_kernel[20] = blue_row_0[x_index];
		red_kernel[15] = red_row_1[x_index];
		green_kernel[15] = green_row_1[x_index];
		blue_kernel[15] = blue_row_1[x_index];
		red_kernel[10] = red_row_2[x_index];
		green_kernel[10] = green_row_2[x_index];
		blue_kernel[10] = blue_row_2[x_index];
		red_kernel[5] = red_row_3[x_index];
		green_kernel[5] = green_row_3[x_index];
		blue_kernel[5] = blue_row_3[x_index];
	}

	// Kimásoljuk az átmeneti regiszterbe a feltöltött szűrő ablakot
	for_copy: for(int i=0; i<FILTER_W*FILTER_H; i++)
	{
		#pragma HLS UNROLL
		red_kernel_tmp[i] = red_kernel[i];
		green_kernel_tmp[i] = green_kernel[i];
		blue_kernel_tmp[i] = blue_kernel[i];
	}

	// Végrehajtjuk a szűrést
	*r_out = median_filter(red_kernel_tmp);
	*g_out = median_filter(green_kernel_tmp);
	*b_out = median_filter(blue_kernel_tmp);

	// Az eredeti pixeleket letároljuk a sorbufferekbe, hogy később tudjuk őket használni
	//A szűrőben a pixelek sorrendje mindegy(a mediánszűrés úgyis a középsőt választja ki)
	if(x_index - (FILTER_W - 1) >= 0)
	{
		if(y_index % 4 == 0)
		{
			red_row_0[x_index - (FILTER_W - 1)] = red_kernel[4];
			green_row_0[x_index - (FILTER_W - 1)] = green_kernel[4];
			blue_row_0[x_index - (FILTER_W - 1)] = blue_kernel[4];
		}

		if(y_index % 4 == 1)
		{
			red_row_1[x_index - (FILTER_W - 1)] = red_kernel[4];
			green_row_1[x_index - (FILTER_W - 1)] = green_kernel[4];
			blue_row_1[x_index - (FILTER_W - 1)] = blue_kernel[4];
		}

		if(y_index % 4 == 2)
		{
			red_row_2[x_index - (FILTER_W - 1)] = red_kernel[4];
			green_row_2[x_index - (FILTER_W - 1)] = green_kernel[4];
			blue_row_2[x_index - (FILTER_W - 1)] = blue_kernel[4];
		}

		if(y_index % 4 == 3)
		{
			red_row_3[x_index - (FILTER_W - 1)] = red_kernel[4];
			green_row_3[x_index - (FILTER_W - 1)] = green_kernel[4];
			blue_row_3[x_index - (FILTER_W - 1)] = blue_kernel[4];
		}
	}

	// A kernelből balra shifttel eltávolírjuk a más nem szükséges pixel infót
	for(int i =FILTER_W*FILTER_H - 1; i > 0; i-- )
	{
		#pragma HLS UNROLL
		red_kernel[i] = red_kernel[i-1];
		green_kernel[i] = green_kernel[i-1];
		blue_kernel[i] = blue_kernel[i-1];
	}

	if(*de_in == 1)
	{
		x_index++;
	}

	// vezérlőjelek kiadása
	*hs_out = *hs_in;
	*vs_out = *vs_in;
	*de_out = *de_in;
	hs_in_prev = *hs_in;
}
