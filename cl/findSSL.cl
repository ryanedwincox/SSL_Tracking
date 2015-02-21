// TODO convert to float4 to use GPU vector arithimitic optimization
// TODO make static parameters constant
#define BLOCK_SIZE 16  // size of local workspace
__kernel void filter_kernel(
        __global uchar * image, //bgr
        __global uchar * filteredImage, //bgr
        int imageWidth,
        int imageHeight,
        int maskSize,
	__global uchar * debugBuffer
    ) {
    int xpos = get_global_id(0);
    int ypos = get_global_id(1);

    int imgPos = (ypos * imageWidth + xpos);

    filteredImage[imgPos] = image[imgPos];

}


