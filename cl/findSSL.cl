#define LOCAL_SIZE 256  // size of local workspace
__kernel void filter_kernel(
        const __global uchar * img, //bgr
        __global uchar * newImg, //bgr
        __local uchar * imgRow,
        const int w,
        const int h,
        const int win,
        const double p,
        volatile __global int * matchesIndex,
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


        double thresh = 0.58;
        if (m1 > thresh)
        {
            newImg[globalPos] = m1 * 255;

            // write corrdinates of matches to matches and increment matches index twice to store corrdinates in the correct spot
            int index = atomic_inc(matchesIndex);

            matches[2*index+0] = core;
            matches[2*index+1] = ypos;
        }
        else if (m2 > thresh)
        {
            newImg[globalPos] = m2 * 255;

            // write corrdinates of matches to matches and increment matches index twice to store corrdinates in the correct spot
            int index = atomic_inc(matchesIndex);

            matches[2*index+0] = core;
            matches[2*index+1] = ypos;
        }
        else
        {
            newImg[globalPos] = 0;
        }
    }
}


