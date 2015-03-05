#include <QApplication>
#include <CL/cl.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stack>

#include "search.h"
#include "/opt/ros/groovy/include/opencv2/video/tracking.hpp"

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
    const char* filename = "/home/pierre/Documents/SSL_Tracking/images/OrcusPortageBayMarker.mp4";
    cv::VideoCapture cap(filename);
    if(!cap.isOpened())  // check if we succeeded
    {
        std::cout << "camera not found" << std::endl;
        return -1;
    }

//    double fps=cap.get(CV_CAP_PROP_FPS);
//    std::cout << fps << std::endl;

    // define kernel files
    const char * findSSLClPath = "/home/pierre/Documents/SSL_Tracking/cl/findSSL.cl";

    search s1;
    search s2;
    cl_int win = 40;
    cl_double p = 0.5;
    s1.buildProgram(findSSLClPath, win, p);
    s2.buildProgram(findSSLClPath, win, p);

    // Create kalman filter
    cv::KalmanFilter KF(4, 2, 0);
    KF.transitionMatrix = *(cv::Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
    cv::Mat_<float> measurement(2,1); measurement.setTo(cv::Scalar(0));

    // Initialize kalman filter
    KF.statePre.at<float>(0) = 0;
    KF.statePre.at<float>(1) = 0;
    KF.statePre.at<float>(2) = 0;
    KF.statePre.at<float>(3) = 0;
    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-4));  // lower values mean more prediction
    setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-3));  // lower values tighten on found points
    setIdentity(KF.errorCovPost, cv::Scalar::all(.1));

    bool firstTime = true;

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
        int blockSize = 31;
        int c = 0;
        cv::Mat imgBin;
        cv::adaptiveThreshold(imgGray, imgBin, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, blockSize, c);

        // transpose for verticle detection
        cv::Mat imgBinTrans;
        cv::transpose(imgBin, imgBinTrans);

        if (firstTime)
        {
            s1.setImage(imgBin);
            s2.setImage(imgBinTrans);
            firstTime = false;
        }

        s1.runProgram(imgBin);
        s2.runProgram(imgBinTrans);
        // newDataPointer is used to display image
        unsigned char* newDataPointer = (unsigned char*) s1.readOutput();
        unsigned char* newDataPointer2 = (unsigned char*) s1.readOutput();
        int matchIndex = s1.readMatchesIndexOutput();
        int matchIndex2 = s2.readMatchesIndexOutput();

        // Create a list to store all matches
        std::list< cv::Point > matches;

//        std::cout << "Match Index X: " << matchIndex << std::endl;
        if (matchIndex > 0)
        {
            unsigned int* newMatchesPointer = s1.readMatchesOutput(matchIndex);

            // Print matches
//            std::cout << "Matches X" << std::endl;
            for (int i = 0; i < matchIndex; i++)
            {

                cv::Point match (newMatchesPointer[2*i], newMatchesPointer[2*i+1]);
                matches.push_front(match);
//                std::cout << "match: " << matches.front() << std::endl;

//                // Color a point at each match
//                cv::circle(img, matches.front(), 3, cv::Scalar(0,255,0), -1);
            }
        }

//        std::cout << "Match Index Y: " << matchIndex2 << std::endl;
        if (matchIndex2 > 0)
        {
            unsigned int* newMatchesPointer = s2.readMatchesOutput(matchIndex2);

            // Print matches
//            std::cout << "Matches Y" << std::endl;
            for (int i = 0; i < matchIndex2; i++)
            {
                cv::Point match (newMatchesPointer[2*i+1], newMatchesPointer[2*i]);
                matches.push_front(match);
//                std::cout << "match: " << matches.front() << std::endl;

//                // Color a point at each match
//                cv::circle(img, matches.front(), 3, cv::Scalar(0,255,0), -1);
            }
        }

        // AVERAGE CLUSTERS
        // Creates a list to store all averaged matches
        std::list< cv::Point > avgMatches;
        while (!matches.empty())
        {
            int xsum = 0;
            int ysum = 0;
            // get current cluster and remove first corrdinate from list
            cv::Point cluster = matches.front();
            matches.pop_front();
            int i = 0;
            int count = 0;
            int radius = 30;
            while (i < matches.size())
            {
                cv::Point match = matches.front();
                if (abs(match.x - cluster.x) < radius && abs(match.y - cluster.y) < radius)
                {
                    matches.pop_front();
                    xsum+= match.x;
                    ysum+= match.y;
                    i--;
                    count++;
                }
                i++;
            }

            // only count matches if there are several in a cluster
//            std::cout << count << std::endl;
            int minClusterSize = 7;
            if (count > minClusterSize)
            {
                cv::Point avgMatch (xsum/count, ysum/count);
                avgMatches.push_front(avgMatch);
            }
        }

//        std::cout << "Cluster center" << std::endl;

//        // Draw red taget over averaged matches
//        for (int i = 0; i < avgMatches.size(); i++)
//        {
//            int l = 10; //radius of cross
//            cv::Point center = avgMatches.front();
////            std::cout << center << std::endl;
//            avgMatches.pop_front();

//            cv::line(img, (cv::Point){center.x-l,center.y}, (cv::Point){center.x+l,center.y}, cv::Scalar(0,0,255), 2);
//            cv::line(img, (cv::Point){center.x,center.y-l}, (cv::Point){center.x,center.y+l}, cv::Scalar(0,0,255), 2);

//        }

        // Run Kalman filter
        // Get target location
        cv::Point center;
        if (avgMatches.size() > 0)
        {
            center = avgMatches.front();
        }
        int l = 10; //radius of cross
        cv::line(img, (cv::Point){center.x-l,center.y}, (cv::Point){center.x+l,center.y}, cv::Scalar(0,0,255), 2);
        cv::line(img, (cv::Point){center.x,center.y-l}, (cv::Point){center.x,center.y+l}, cv::Scalar(0,0,255), 2);

        static int numPredictions = 0;
        static int maxNumPredictions = 20;

        std::cout << numPredictions << std::endl;

        // No match found
        if (avgMatches.size() == 0 && numPredictions < maxNumPredictions)
        {
            center.x = KF.statePre.at<float>(0);
            center.y = KF.statePre.at<float>(1);
            numPredictions++;
        }
        if (avgMatches.size() != 0)  // match found
        {
            avgMatches.pop_front();
            numPredictions = 0;
        }

        if (avgMatches.size() != 0 || numPredictions < maxNumPredictions)
        {
            // First predict, to update the internal statePre variable
            cv::Mat prediction = KF.predict();
            cv::Point predictPt(prediction.at<float>(0),prediction.at<float>(1));

            measurement(0) = center.x;
            measurement(1) = center.y;

            // The "correct" phase that is going to use the predicted value and our measurement
            cv::Mat estimated = KF.correct(measurement);
            cv::Point statePt(estimated.at<float>(0),estimated.at<float>(1));

            // draw blue cross at kalman filter estimation
            l = 10; //radius of cross
            cv::line(img, (cv::Point){statePt.x-l,statePt.y}, (cv::Point){statePt.x+l,statePt.y}, cv::Scalar(255,0,0), 2);
            cv::line(img, (cv::Point){statePt.x,statePt.y-l}, (cv::Point){statePt.x,statePt.y+l}, cv::Scalar(255,0,0), 2);
        }


        // newImage is passed into the next filter
        cv::Mat newImage = cv::Mat(cv::Size(w,h), CV_8UC1, newDataPointer2);

        // Display images
        cv::imshow("New Image", newImage);

//        cv::imshow("Binary Image", imgBin);

        cv::imshow("Original Image", img);

        // keep window open until any key is pressed
//        cv::waitKey(1);
        if(cv::waitKey(1) >= 0) break;
//        break;
    }
}
