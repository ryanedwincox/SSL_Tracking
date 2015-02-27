#include "search.h"

using namespace std;

// Constructor
search::search()
{
    cl_int init_status;
    cl_platform_id *platforms;

    cl_uint numPlatforms = 0;

    // Query for the number of recongnized platforms
    init_status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if(init_status != CL_SUCCESS && VERBOSE) {
        cout << "GPU init: clGetPlatformIDs failed" << endl;
    }

    // Make sure some platforms were found
    if(numPlatforms == 0 && VERBOSE) {
        cout << "GPU init: No platforms detected" << endl;
    }

    // Allocate enough space for each platform
    platforms = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));
    if(platforms == NULL && VERBOSE) {
        cout << "not enough space for platforms" << endl;
    }

    // Fill in platforms
    init_status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    if(init_status != CL_SUCCESS && VERBOSE) {
        cout << "GPU init: clGetPlatformIDs failed" << endl;
    }

    // Print out some basic information about each platform
    if (VERBOSE)
    {
        cout << numPlatforms << " platforms detected" << endl;
        for(unsigned int i = 0; i < numPlatforms; i++) {
            char buf[100];
            cout << "Platform: " << i << endl;
            init_status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR,
                                            sizeof(buf), buf, NULL);
            cout << "\tVendor: " << buf << endl;
            init_status |= clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME,
                                             sizeof(buf), buf, NULL);
            cout << "\tName: " <<  buf << endl;

            if(init_status != CL_SUCCESS) {
                cout << "GPU init: clGetPlatformInfo failed" << endl;
            }
        }
    }

    // get device count
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount);
    if (VERBOSE)
    {
        cout << "device count error: " << err << endl;
    }

    // get devices
    devices = new cl_device_id[deviceCount];
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, deviceCount, devices, NULL);
    if (VERBOSE)
    {
        cout << "device ID error: " << err << endl;
    }

    // create a single context for all devices
    context = clCreateContext(NULL, deviceCount, devices, NULL, NULL, &err);
    if (VERBOSE)
    {
        cout << "context error: " << err << endl;
    }
}

// Builds OpenCl program
void search::buildProgram(const char* clPath, cl_int win, cl_double p)
{
    this->clPath = clPath;
    this->win = win;
    this->p = p;

    // get size of kernel source
    const char* kernelSource = clPath;
    programHandle = fopen(kernelSource, "r");
    if (programHandle == NULL && VERBOSE)
    {
        cout << "Kernel source not found" << endl;
    }
    fseek(programHandle, 0, SEEK_END);
    programSize = ftell(programHandle);
    rewind(programHandle);

    // read kernel source into buffer
    programBuffer = (char*) malloc(programSize + 1);
    programBuffer[programSize] = '\0';
    fread(programBuffer, sizeof(char), programSize, programHandle);
    fclose(programHandle);

    // create program from buffer
    program = clCreateProgramWithSource(context, 1,
            (const char**) &programBuffer, &programSize, NULL);

    // build program
    const char* buildOptions = "";
    err = clBuildProgram(program, deviceCount, devices, buildOptions, NULL, NULL);
    if (VERBOSE)
    {
        cout << "program error: " << err << "\n";
    }

    // create the log string and show it to the user. Then quit
    char buildLog[MAX_LOG_SIZE];
    err = clGetProgramBuildInfo(program,
                          devices[0],
                          CL_PROGRAM_BUILD_LOG,
                          MAX_LOG_SIZE,
                          &buildLog,
                          NULL);
    if (VERBOSE)
    {
        printf("**BUILD LOG**\n%s",buildLog);
        cout << "clGetProgramBuildInfo error: " << err << "\n";
    }

    //create queue to which we will push commands for the device
    queue = clCreateCommandQueue(context,devices[0],0,&err);
    if (VERBOSE)
    {
        cout << "command queue error: " << err << "\n";
    }

    // build kernel
    kernel = clCreateKernel(program, "filter_kernel", &err);
    if (VERBOSE)
    {
        cout << "cl_kernel error: " << err << "\n";
    }
}

// Stores image to process
// Creates buffers to store image on device
void search::setImage(cv::Mat img)
{
    if (VERBOSE)
    {
        cout << "Creating image buffers" << endl;
    }

    image = img;

    imageWidth = image.cols;
    imageHeight = image.rows;
    imageSize = imageHeight * imageWidth;

    // Create an OpenCL buffer for the image
    clImage = clCreateBuffer(context,
                             CL_MEM_READ_ONLY,
                             imageSize * sizeof(char),
                             NULL,
                             &err);
    if (VERBOSE)
    {
        cout << "clImage Buffer error: " << err << "\n";
    }

    // Create an OpenCL buffer for the result
    clResult = clCreateBuffer(context,
                              CL_MEM_WRITE_ONLY,
                              imageSize * sizeof(char),
                              NULL,
                              &err);
    if (VERBOSE)
    {
        cout << "clResult Buffer error: " << err << "\n";
    }

    // Create matches buffer
    clMatchIndex = clCreateBuffer(context,
                             CL_MEM_WRITE_ONLY,
                             sizeof(cl_int),
                             NULL,
                             &err);
    if (VERBOSE)
    {
        cout << "clMatchIndex Buffer error: " << err << "\n";
    }

    if (VERBOSE)
    {
        cout << "clMatchIndex Buffer error: " << err << "\n";
    }

    // Create matches buffer
    clMatch = clCreateBuffer(context,
                             CL_MEM_WRITE_ONLY,
                             MATCHES_BUFFER_SIZE * sizeof(int),
                             NULL,
                             &err);
    if (VERBOSE)
    {
        cout << "clMatch Buffer error: " << err << "\n";
    }

    // load image to device
    err = clEnqueueWriteBuffer(queue,
                               clImage,
                               CL_TRUE,
                               0,
                               imageSize * sizeof(char),
                               (void*) &image.data[0],
                               0,
                               NULL,
                               NULL);
    if (VERBOSE)
    {
        cout << "enqueueWriteBuffer image error: " << err << "\n";
    }

    // set matchIndex to 0
    matchesIndex = 0;
    err = clEnqueueWriteBuffer(queue,
                               clMatchIndex,
                               CL_TRUE,
                               0,
                               sizeof(cl_int),
                               (void*) &matchesIndex,
                               0,
                               NULL,
                               NULL);
    if (VERBOSE)
    {
        cout << "enqueueWriteBuffer matchIndex error: " << err << "\n";
    }
}

// Excecutes the kernel
void search::runProgram()
{
    if (VERBOSE)
    {
        std::cout << "runProgram" << std::endl;
    }

    unsigned int argnum = 0;
    // set kernel arguments
    err = clSetKernelArg(kernel, argnum++, sizeof(cl_mem), (void *)&clImage);
    if (VERBOSE)
    {
        cout << "kernel arg 0 error: " << err << "\n";
    }
    err = clSetKernelArg(kernel, argnum++, sizeof(cl_mem), (void *)&clResult);
    if (VERBOSE)
    {
        cout << "kernel arg 1 error: " << err << "\n";
    }
    err = clSetKernelArg(kernel, argnum++, (size_t)imageWidth*sizeof(uchar), NULL);
    if (VERBOSE)
    {
        cout << "kernel arg 2 error: " << err << "\n";
    }
    err = clSetKernelArg(kernel, argnum++, sizeof(cl_int), &imageWidth);
    if (VERBOSE)
    {
        cout << "kernel arg 3 error: " << err << "\n";
    }
    err = clSetKernelArg(kernel, argnum++, sizeof(cl_int), &imageHeight);
    if (VERBOSE)
    {
        cout << "kernel arg 4 error: " << err << "\n";
    }
    err = clSetKernelArg(kernel, argnum++, sizeof(cl_int), &win);
    if (VERBOSE)
    {
        cout << "kernel arg 5 error: " << err << "\n";
    }
    err = clSetKernelArg(kernel, argnum++, sizeof(cl_double), &p);
    if (VERBOSE)
    {
        cout << "kernel arg 6 error: " << err << "\n";
    }
    err = clSetKernelArg(kernel, argnum++, sizeof(cl_mem), &clMatchIndex);
    if (VERBOSE)
    {
        cout << "kernel arg 7 error: " << err << "\n";
    }
    err = clSetKernelArg(kernel, argnum++, sizeof(cl_mem), &clMatch);
    if (VERBOSE)
    {
        cout << "kernel arg 8 error: " << err << "\n";
    }
//    CL_SUCCESS

    // Set local and global workgroup sizes
    size_t localws[2] = {256,1};
    size_t globalws[2] = {256, imageHeight};

    // Run filter kernel
    err = clEnqueueNDRangeKernel(queue,
                                 kernel,
                                 2,
                                 NULL,
                                 globalws,
                                 localws,
                                 0,
                                 NULL,
                                 NULL);
    if (VERBOSE)
    {
        cout << "clEnqueueNDRangeKernel error: " << err << "\n";
    }
//    CL_SUCCESS

//    clImage = clResult;  // allows the image to be processed multiiple times
}

// Returns the data read from the output buffer
void* search::readOutput() {
    if (VERBOSE)
    {
        std::cout << "readOutput" << std::endl;
    }

    unsigned char newData [imageSize * 2 * sizeof(char)];  // **** For some reason making this double the needed size get rid of artifacts at the bottom of displayed image

    // Transfer image back to host
    err = clEnqueueReadBuffer(queue,
                              clResult,
                              CL_TRUE,
                              0,
                              imageSize * sizeof(char),
                              (void*) newData,
                              0,
                              NULL,
                              NULL);
    if (VERBOSE)
    {
        cout << "enqueueReadImage error: " << err << "\n";
    }

    return newData;
}

// Transfer Match buffer back to host
unsigned int* search::readMatchesOutput(unsigned int numMatches)
{
    unsigned int matches [MATCHES_BUFFER_SIZE * sizeof(cl_int)];
    err = clEnqueueReadBuffer(queue,
                              clMatch,
                              CL_TRUE,
                              0,
                              (size_t)numMatches * 2 * sizeof(cl_int), // 2 elements per match
                              matches,
                              0,
                              NULL,
                              NULL);
    if (VERBOSE)
    {
        cout << "clMatch read buffer error: " << err << "\n";
    }

    return matches;
}

// Transfer matchesIndex buffer back to host
int search::readMatchesIndexOutput()
{
    int index;
    err = clEnqueueReadBuffer(queue,
                              clMatchIndex,
                              CL_TRUE,
                              0,
                              sizeof(int),
                              &index,
                              0,
                              NULL,
                              NULL);
    if (VERBOSE)
    {
        cout << "clMatchIndex read buffer error: " << err << "\n";
    }

    return index;
}

cv::Mat search::getInputImage()
{
    return image;
}

search::~search()
{
}

