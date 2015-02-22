// TODO convert to float4 to use GPU vector arithimitic optimization
// TODO make static parameters constant
#define BLOCK_SIZE 16  // size of local workspace
__kernel void filter_kernel(
        __global uchar * img, //bgr
        __global uchar * filteredImage, //bgr
        int w,
        int h,
        int maskSize,
	__global uchar * debugBuffer
    ) {
    int xpos = get_global_id(0);
    int ypos = get_global_id(1);

    int imgPos = (ypos * w + xpos);

    int sumMatch = 0;
    int sumMismatch = 0;
    int win = 50; // TODO make this a param
    double p = 0.5; // TODO make this a param
    for (int i = 0; i < win; i++)
    {
        if (xpos+i < w)
        {
            int ip = i*p;
            sumMatch = sumMatch + abs(img[imgPos+i] - img[(int)imgPos+ip]) / 255;
            int irootp = i*0.7071;
            sumMismatch = sumMismatch + abs(img[imgPos+i] - img[imgPos+irootp]) / 255;
        }
    }

    double m = (double)(sumMismatch - sumMatch) / win; // matching function value

    if (m > 0.6)
    {
        filteredImage[imgPos] = m * 255;
    }
    else
    {
        filteredImage[imgPos] = 0;
    }


}


