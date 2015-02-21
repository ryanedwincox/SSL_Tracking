// TODO convert to float4 to use GPU vector arithimitic optimization
// TODO make static parameters constant
#define BLOCK_SIZE 16  // size of local workspace
__kernel void filter_kernel(
        __global uchar4 * image, //bgr
        __global uchar4 * filteredImage, //bgr
        int imageWidth,
        int imageHeight,
        int maskSize,
	__global uchar * debugBuffer
    ) {
    // storing image data locally
    __local uchar4 P[BLOCK_SIZE*BLOCK_SIZE];
    // Identify workgroup
    int i = get_group_id(0);
    int j = get_group_id(1);
    // Indentify workitem
    int iDx = get_local_id(0);
    int iDy = get_local_id(1);
    int ii = i*BLOCK_SIZE + iDx;  // == get_global_id(0)
    int jj = j*BLOCK_SIZE + iDy;  // == get_global_id(1)

    int imgPosGlobal = (jj * imageWidth + ii);
    int imgPosLocal = (iDy * BLOCK_SIZE + iDx);

    // Read pixels
    P[imgPosLocal] = image[imgPosGlobal];
    barrier(CLK_LOCAL_MEM_FENCE);

    filteredImage[imgPosGlobal] = P[imgPosLocal];

    // This is cool, it shows each workgroup
    //filteredImage[imgPosGlobal+0] = P[0];
    //filteredImage[imgPosGlobal+1] = P[1];
    //filteredImage[imgPosGlobal+2] = P[2];

}

//// TODO make static parameters constant
//__kernel void filter_kernel(
//        __global uchar4 * image, //bgr
//        __global uchar4 * filteredImage, //bgr
//        int imageWidth,
//        int imageHeight,
//        int maskSize,
//	__global uchar * debugBuffer
//    ) {
//    int xpos = get_global_id(0);
//    int ypos = get_global_id(1);

//    int imgPos = (ypos * imageWidth + xpos);

//    filteredImage[imgPos] = image[imgPos];
//}
