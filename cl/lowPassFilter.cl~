__kernel void filter_kernel(
        __global uchar * image, //bgr
        __global uchar4 * filteredImage, //bgra
	int imageWidth,
	int imageHeight,
        __private int maskSize,
	__global uchar * debugBuffer
    ) {
    int xpos = get_global_id(0);
    int ypos = get_global_id(1); 

    int imgPos = (ypos * imageWidth + xpos) * 3;
    int filImgPos = (ypos * imageWidth + xpos);
    filteredImage[filImgPos].x = image[imgPos+0];
    filteredImage[filImgPos].y = image[imgPos+1];
    filteredImage[filImgPos].z = image[imgPos+2];
    filteredImage[filImgPos].w = 100;

}


