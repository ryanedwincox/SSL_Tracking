// TODO convert to float4 to use GPU vector arithimitic optimization
// TODO make static parameters constant
#define LOCAL_SIZE 256  // size of local workspace
__kernel void filter_kernel(
        const __global uchar * img, //bgr
        __global uchar * newImg, //bgr
        __local uchar * imgRow,
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

    // Copy a row into local memory
    event_t e = async_work_group_copy (imgRow, &img[j*w], (size_t) w, (event_t) 0);

    // Make sure all threads have finished loading all pixels
    wait_group_events (1, &e);

    // This loop spreads the work for the whole row onto one local workspace
    for (int core = iDx; core < w; core+=LOCAL_SIZE)
    {

        int globalPos = (ypos * w + core);

        int sumMatch1 = 0;
        int sumMismatch1 = 0;
        int sumMatch2 = 0;
        int sumMismatch2 = 0;
        for (int i = 10; i < win; i++)
        {
            if (xpos+i <= w)
            {
                int ip = i*p;
                sumMatch1 = sumMatch1 + abs(imgRow[core+i] - imgRow[core+ip]) / 255;
                int irootp = i*sqrt(p);
                sumMismatch1 = sumMismatch1 + abs(imgRow[core+i] - imgRow[core+irootp]) / 255;
            }
            if (xpos-i >= 0)
            {
                int ip = i*p;
                sumMatch2 = sumMatch2 + abs(imgRow[core-i] - imgRow[core-ip]) / 255;
                int irootp = i*sqrt(p);
                sumMismatch2 = sumMismatch2 + abs(imgRow[core-i] - imgRow[core-irootp]) / 255;
            }
        }

        double m1 = (double) (sumMismatch1 - sumMatch1) / win; // matching function value
        double m2 = (double) (sumMismatch2 - sumMatch2) / win; // matching function value

        if (m1 > 0.5)
        {
            newImg[globalPos] = m1 * 255;
    //        matchesIndex++;
    //        matches[matchesIndex] = xpos;
    //        matches[matchesIndex + 1] = ypos;
        }
        else if (m2 > 0.5)
        {
            newImg[globalPos] = m2 * 255;
        }
        else
        {
            newImg[globalPos] = 0;
        }


//         // copy image
//        newImg[globalPos] = imgRow[core];
    }
}


