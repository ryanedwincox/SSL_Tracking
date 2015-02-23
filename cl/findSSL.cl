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

    int sumMatch1 = 0;
    int sumMismatch1 = 0;
    int sumMatch2 = 0;
    int sumMismatch2 = 0;
    for (int i = 0; i < win; i++)
    {
        if (xpos+i <= w)
        {
            int ip = i*p;
            sumMatch1 = sumMatch1 + abs(img[imgPos+i] - img[(int)imgPos+ip]) / 255;
            int irootp = i*0.7071;
            sumMismatch1 = sumMismatch1 + abs(img[imgPos+i] - img[imgPos+irootp]) / 255;
        }
        if (xpos-i >= 0)
        {
            int ip = i*p;
            sumMatch2 = sumMatch2 + abs(img[imgPos-i] - img[(int)imgPos-ip]) / 255;
            int irootp = i*0.7071;
            sumMismatch2 = sumMismatch2 + abs(img[imgPos-i] - img[imgPos-irootp]) / 255;
        }
    }

    double m1 = (double) (sumMismatch1 - sumMatch1) / win; // matching function value
    double m2 = (double) (sumMismatch2 - sumMatch2) / win; // matching function value

    if (m1 > 0.6)
    {
        newImg[imgPos] = m1 * 255;
//        matchesIndex++;
//        matches[matchesIndex] = xpos;
//        matches[matchesIndex + 1] = ypos;
    }
    else if (m2 > 0.6)
    {
        newImg[imgPos] = m2 * 255;
    }
    else
    {
        newImg[imgPos] = 0;
    }


}


