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
    search s2;
    cl_int win = 40;
    cl_double p = 0.5;
    s1.buildProgram(findSSLClPath, win, p);
    s2.buildProgram(findSSLClPath, win, p);

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
        double thresh = 110;
        cv::Mat imgBin;
        cv::threshold(imgGray, imgBin, thresh, 255, cv::THRESH_BINARY);

        // transpose for verticle detection
        cv::Mat imgBinTrans;
        cv::transpose(imgBin, imgBinTrans);

        s1.setImage(imgBin);
        s2.setImage(imgBinTrans);

        s1.runProgram();
        s2.runProgram();
        // newDataPointer is used to display image
        unsigned char* newDataPointer = (unsigned char*) s1.readOutput();
        unsigned char* newDataPointer2 = (unsigned char*) s1.readOutput();
        int matchIndex = s1.readMatchesIndexOutput();
        int matchIndex2 = s2.readMatchesIndexOutput();

        std::cout << "Match Index X: " << matchIndex << std::endl;
        if (matchIndex > 0)
        {
            unsigned int* newMatchesPointer = s1.readMatchesOutput(matchIndex);

            // Print matches
            std::cout << "Matches X" << std::endl;
            for (int i = 0; i < matchIndex; i++)
            {
                int x = newMatchesPointer[2*i];
                int y = newMatchesPointer[2*i+1];
                std::cout << "match: " << x << "," << y << std::endl;

                cv::circle(img, cv::Point(x,y), 3, cv::Scalar(0,255,0), -1);
            }
        }

        std::cout << "Match Index Y: " << matchIndex2 << std::endl;
        if (matchIndex2 > 0)
        {
            unsigned int* newMatchesPointer = s2.readMatchesOutput(matchIndex2);

            // Print matches
            std::cout << "Matches Y" << std::endl;
            for (int i = 0; i < matchIndex2; i++)
            {
                int x = newMatchesPointer[2*i+1];
                int y = newMatchesPointer[2*i];
                std::cout << "match: " << x << "," << y << std::endl;

                cv::circle(img, cv::Point(x,y), 3, cv::Scalar(0,255,0), -1);
            }
        }

        // newImage is passed into the next filter
        cv::Mat newImage = cv::Mat(cv::Size(w,h), CV_8UC1, newDataPointer2);

        // Display images
        cv::imshow("New Image", newImage);

        cv::imshow("Binary Image", imgBin);

        cv::imshow("Original Image", img);

        // keep window open until any key is pressed
//        cv::waitKey(1);
        if(cv::waitKey(1) >= 0) break;
//        break;
    }
}
