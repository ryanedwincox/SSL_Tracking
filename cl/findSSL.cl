// TODO convert to float4 to use GPU vector arithimitic optimization
// TODO make static parameters constant
#define BLOCK_SIZE 16  // size of local workspace
__kernel void filter_kernel(
        __global uchar * img, //bgr
        __global uchar * newImg, //bgr
        int w,
        int h,
        int win,
        double p,
        __global int * matches
    ) {
    int xpos = get_global_id(0);
    int ypos = get_global_id(1);

    int imgPos = (ypos * w + xpos);

//    __global int matchesIndex = 0;

    int sumMatch = 0;
    int sumMismatch = 0;
//    int win = 50; // TODO make this a param
//    double p = 0.5; // TODO make this a param
    for (int i = 0; i < win; i++)
    {
        if (xpos+i <= w)
        {
            int ip = i*p;
            sumMatch = sumMatch + abs(img[imgPos+i] - img[(int)imgPos+ip]) / 255;
            int irootp = i*0.7071;
            sumMismatch = sumMismatch + abs(img[imgPos+i] - img[imgPos+irootp]) / 255;
        }
        if (xpos-i >= 0)
        {
            int ip = i*p;
            sumMatch = sumMatch + abs(img[imgPos-i] - img[(int)imgPos-ip]) / 255;
            int irootp = i*0.7071;
            sumMismatch = sumMismatch + abs(img[imgPos-i] - img[imgPos-irootp]) / 255;
        }
    }

    double m = (double) (sumMismatch - sumMatch) / win / 2; // matching function value

    if (m > 0.6)
    {
        newImg[imgPos] = m * 255;
//        matchesIndex++;
//        matches[matchesIndex] = xpos;
//        matches[matchesIndex + 1] = ypos;
    }
    else
    {
        newImg[imgPos] = 0;
    }


}


