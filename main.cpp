#include <QApplication>
#include <CL/cl.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include "search.h"

int main(int argc, char *argv[])
{
//    // Load image
//    cv::Mat img = cv::imread("/home/pierre/Documents/SSL_Tracking/images/marker_sample_1.jpg", CV_LOAD_IMAGE_COLOR);

//    // Check for invalid input
//    if(! img.data )
//    {
//        std::cout <<  "Could not open or find the image" << std::endl;
//    }
//    else
//    {
//        std::cout << "Image loaded" << std::endl;
//    }

//    int w = img.cols;
//    int h = img.rows;
//    std::cout << "image width: " << w << " image height: " << h << std::endl;

//    // Resize image before any other processing
//    cv::resize(img, img, cv::Size(w/4,h/4), 0.25, 0.25, cv::INTER_AREA);
//    // update w and h to new image size
//    w = img.cols;
//    h = img.rows;
//    std::cout << "image width: " << w << " image height: " << h << std::endl;

    //Create video capture object
    int cameraNum = 0;
    cv::VideoCapture cap(cameraNum);
    if(!cap.isOpened())  // check if we succeeded
    {
        std::cout << "camera not found" << std::endl;
        return -1;
    }

    // define kernel files
    const char * findSSLClPath = "/home/pierre/Documents/SSL_Tracking/cl/findSSL.cl";

    search s1;
    cl_int win = 40;
    cl_double p = 0.5;
    s1.buildProgram(findSSLClPath, win, p);

    cv::namedWindow("Original Image", cv::WINDOW_AUTOSIZE); // Create a window for display.

    while (cap.isOpened())
    {
        // Get new frame
        cv::Mat img;
        cap >> img;

        int w = img.cols;
        int h = img.rows;
        if (VERBOSE)
        {
            std::cout << "image width: " << w << " image height: " << h << std::endl;
        }

        // convert to grayscale
        cv::Mat imgGray;
        cvtColor(img, imgGray, CV_BGR2GRAY);

        // convert to binary
        double thresh = 150;
        cv::Mat imgBin;
        cv::threshold(imgGray, imgBin, thresh, 255, cv::THRESH_BINARY);

        s1.setImage(imgBin);

        s1.runProgram();
        // newDataPointer is used to display image in gui
        unsigned char* newDataPointer = (unsigned char*) s1.readOutput();
        unsigned int* newMatchesPointer = (unsigned int*) s1.readMatchesOutput();
        // newImage is passed into the next filter
        cv::Mat newImage = cv::Mat(cv::Size(w,h), CV_8UC1, newDataPointer);

//        // Print matches
//        for (int i = 0; i < MATCHES_BUFFER_SIZE; i+=2)
//        {
//            std::cout << "match: " << newMatchesPointer[i] << "," << newMatchesPointer[i+1] << std::endl;
//        }

//        // Display images
        cv::imshow("New Image", newImage);           // Show our image inside it.

        cv::imshow("Original Image", imgBin);           // Show our image inside it.

        // keep window open until any key is pressed
//        cv::waitKey(1);
        if(cv::waitKey(1) >= 0) break;
//        break;
    }
}
