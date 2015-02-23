// TODO convert to float4 to use GPU vector arithimitic optimization
// TODO make static parameters constant
#define LOCAL_SIZE 256  // size of local workspace
__kernel void filter_kernel(
        const __global uchar * img, //bgr
        __global uchar * newImg, //bgr
        int w,
        int h,
        int win,
        double p,
        __global int * matches
    ) {
    // Identify workgroup
    int i = get_group_id(0);
    int j = get_group_id(1);

    // Indentify workitem
    int iDx = get_local_id(0);
    int iDy = get_local_id(1);
    int xpos = i * LOCAL_SIZE + iDx;  // == get_global_id(0)
    int ypos = j + iDy;  // == get_global_id(1)

    int imgPos = (ypos * w + xpos);

    // storing image data locally
    // TODO pass in this memory as a paramter to allow for variable sizing
    __local uchar imgRow [640];  // cannot be variable values

//    // Copy a row into local memory
//    int numEvents = 1;
//    event_t e = (event_t) 0;
//    size_t ww = 640;  // *************************
//    e = async_work_group_copy ((__local uchar*)&imgRow, (const __global uchar *)&(img[j]), ww, e);

//    // Make sure all threads have finished loading all pixels
//    wait_group_events (numEvents, &e);

//    if (iDx < 200)
//    {
    imgRow[iDx] = img[imgPos];
//    }

    // Make sure all threads have finished loading all pixels
    barrier(CLK_LOCAL_MEM_FENCE);

    int sumMatch1 = 0;
    int sumMismatch1 = 0;
    int sumMatch2 = 0;
    int sumMismatch2 = 0;
    for (int i = 0; i < win; i++)
    {
        if (xpos+i <= w)
        {
            int ip = i*p;
            sumMatch1 = sumMatch1 + abs(imgRow[iDx+i] - imgRow[iDx+ip]) / 255;
            int irootp = i*sqrt(p);
            sumMismatch1 = sumMismatch1 + abs(imgRow[iDx+i] - imgRow[iDx+irootp]) / 255;
        }
        if (xpos-i >= 0)
        {
            int ip = i*p;
            sumMatch2 = sumMatch2 + abs(imgRow[iDx-i] - imgRow[iDx-ip]) / 255;
            int irootp = i*sqrt(p);
            sumMismatch2 = sumMismatch2 + abs(imgRow[iDx-i] - imgRow[iDx-irootp]) / 255;
        }
    }

    double m1 = (double) (sumMismatch1 - sumMatch1) / win; // matching function value
    double m2 = (double) (sumMismatch2 - sumMatch2) / win; // matching function value


//    newImg[imgPos] = imgRow[iDx];


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


