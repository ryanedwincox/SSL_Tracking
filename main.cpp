#include <QApplication>
#include <CL/cl.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stack>

#include "search.h"
#include "/opt/ros/groovy/include/opencv2/video/tracking.hpp"

// declare local methods
cv::KalmanFilter createKalmanFilter(int x, int y);

int main(int argc, char *argv[])
{
    //Create video capture object
    int cameraNum = 0;
    const char* filename = "/home/pierre/Documents/SSL_Tracking/images/OrcusPortageBayMarker.mp4";
    cv::VideoCapture cap(cameraNum);
    if(!cap.isOpened())  // check if we succeeded
    {
        std::cout << "camera not found" << std::endl;
        return -1;
    }

//    double fps=cap.get(CV_CAP_PROP_FPS);
//    std::cout << fps << std::endl;de

    // define kernel files
    const char * findSSLClPath = "/home/pierre/Documents/SSL_Tracking/cl/findSSL.cl";

    // Initialize OpenCL
    search s1;
    search s2;
    cl_int win = 40;
    cl_double p = 0.5;
    s1.buildProgram(findSSLClPath, win, p);
    s2.buildProgram(findSSLClPath, win, p);

//    // Create kalman filter
    cv::KalmanFilter KF = createKalmanFilter(0,0);
    cv::Point statePt;

    // firstTime is used to insure the image buffers are only created once
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
        cv::Mat imgBinVert;
        cv::transpose(imgBin, imgBinVert);

        if (firstTime)
        {
            s1.setImage(imgBin);
            s2.setImage(imgBinVert);
            firstTime = false;
        }

        // Run OpenCV kernel to find markers
        s1.runProgram(imgBin);
        s2.runProgram(imgBinVert);
        // newDataPointer is used to display image
        unsigned char* newDataPointer = (unsigned char*) s1.readOutput();
        unsigned char* newDataPointer2 = (unsigned char*) s1.readOutput();
        int matchIndex = s1.readMatchesIndexOutput();
        int matchIndex2 = s2.readMatchesIndexOutput();

        // Create a list to store all matches
        std::list< cv::Point > matches;

        // Read all matches from OpenCL kernel
        if (matchIndex > 0)
        {
            unsigned int* newMatchesPointer = s1.readMatchesOutput(matchIndex);

            // Print matches
            for (int i = 0; i < matchIndex; i++)
            {

                cv::Point match (newMatchesPointer[2*i], newMatchesPointer[2*i+1]);
                matches.push_front(match);

//                // Color a point at each match
//                cv::circle(img, matches.front(), 3, cv::Scalar(0,255,0), -1);
            }
        }

        // Read all matches from OpenCL kernel
        if (matchIndex2 > 0)
        {
            unsigned int* newMatchesPointer = s2.readMatchesOutput(matchIndex2);

            // Print matches
            for (int i = 0; i < matchIndex2; i++)
            {
                cv::Point match (newMatchesPointer[2*i+1], newMatchesPointer[2*i]);
                matches.push_front(match);

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

            // Compare all remaining matches and if they are close to the current match then they are in the same cluster
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
            int minClusterSize = 7;
            if (count > minClusterSize)
            {
                cv::Point avgMatch (xsum/count, ysum/count);
                avgMatches.push_front(avgMatch);
            }
        }

        // Draw red taget over averaged matches
        std::list< cv::Point > avgMatchesCp = avgMatches;
        for (int i = 0; i < avgMatchesCp.size(); i++)
        {
            int l = 10; //radius of cross
            cv::Point center = avgMatchesCp.front();
//            std::cout << center << std::endl;
            avgMatchesCp.pop_front();

            cv::line(img, (cv::Point){center.x-l,center.y}, (cv::Point){center.x+l,center.y}, cv::Scalar(0,0,255), 2);
            cv::line(img, (cv::Point){center.x,center.y-l}, (cv::Point){center.x,center.y+l}, cv::Scalar(0,0,255), 2);
        }

        // Run Kalman filter

        // numPredictions counts the number of frames since the last positive match
        static int numPredictions = 0;
        // maxNumPredictions is the number of frames the filter will guess a position since the last positive match
        static int maxNumPredictions = 20;

        // Get target location if a match was found
        cv::Point center;
        if (avgMatches.size() > 0)
        {
            int radius = 20;


                // loop through all average matches to see if any are close to previous match
                std::list< cv::Point > avgMatchesCopy = avgMatches;
                for (int i = 0; i < avgMatchesCopy.size(); i++)
                {
//                    std::cout << "search" << std::endl;
    //                std::cout << statePt << std::endl;
                    // find if close to previous match
                    if (abs(avgMatchesCopy.front().x - statePt.x) < radius && abs(avgMatchesCopy.front().y - statePt.y) < radius)
                    {
                        // it is close to one of the previous matches so use that one
                        // push it on front of avgMatches so it will be used first
                        // this means there is an extra duplicate match is avgMatches
                        avgMatches.push_front(avgMatchesCopy.front());
//                        std::cout << "found" << std::endl;
                        break;
                    }
//

                    // check all matches
                    avgMatchesCopy.pop_front();
                }

            // current match is close to previous match
            if (abs(avgMatches.front().x - statePt.x) < radius && abs(avgMatches.front().y - statePt.y) < radius)
            {
                center = avgMatches.front();
                avgMatches.pop_front();
                numPredictions = 0;
                std::cout << "case 1" << std::endl;
            }
//            There was no match close to previous match, but the predition is close so use that
            else if (abs(KF.statePre.at<float>(0) - statePt.x) < radius && abs(KF.statePre.at<float>(1) - statePt.y) < radius && numPredictions < maxNumPredictions )// && KF.statePre.at<float>(0) != 0)
            {
                center = (cv::Point){(int)KF.statePre.at<float>(0),(int)KF.statePre.at<float>(1)};
                avgMatches.pop_front();
                numPredictions++;
                std::cout << "case 2" << std::endl;
            }
            // no current matches are close to previous match so create new filter
            else// (abs(avgMatches.front().x - statePt.x) > radius && abs(avgMatches.front().y - statePt.y) > radius && !(numPredictions < maxNumPredictions))
            {
                KF = createKalmanFilter(avgMatches.front().x, avgMatches.front().y);
                center = avgMatches.front();
                avgMatches.pop_front();
                numPredictions = 0;
                std::cout << "case 3" << std::endl;
                std::cout << "new filter***********" << std::endl;
            }

//                std::cout << statePt << std::endl;
//                std::cout << KF.statePre.at<float>(0) << "," << KF.statePre.at<float>(1) << std::endl;
//            // if no current match is in the radius and the predicted value is not in the radius create a new filter
//            if (abs(avgMatches.front().x - statePt.x) > radius && abs(avgMatches.front().y - statePt.y) > radius && !(abs(KF.statePre.at<float>(0) - statePt.x) < radius && abs(KF.statePre.at<float>(1) - statePt.y) < radius))
//            {
//                KF = createKalmanFilter(avgMatches.front().x, avgMatches.front().y);
//                std::cout << "new filter***********" << std::endl;
//                std::cout << KF.statePre.at<float>(0) << std::endl;
//            }
////            if (abs(KF.statePre.at<float>(0) - statePt.x) < radius && abs(KF.statePre.at<float>(1) - statePt.y) < radius)
////            {
////                center = (cv::Point){(int)KF.statePre.at<float>(0),(int)KF.statePre.at<float>(1)};
////            }
////            else
////            {
//                center = avgMatches.front();
//                avgMatches.pop_front();
//                numPredictions = 0;
////            }


            // If a match was found draw it
            int l = 10; //radius of cross
            cv::line(img, (cv::Point){center.x-l,center.y}, (cv::Point){center.x+l,center.y}, cv::Scalar(0,0,255), 2);
            cv::line(img, (cv::Point){center.x,center.y-l}, (cv::Point){center.x,center.y+l}, cv::Scalar(0,0,255), 2);
        }
        // No match found
        // Predict position based on last prediction, don't do this more than maxNumPredictions times
        else if (numPredictions < maxNumPredictions)
        {
            center = (cv::Point){(int)KF.statePre.at<float>(0),(int)KF.statePre.at<float>(1)};
            numPredictions++;

//            int radius = 40;
//            if (abs(center.x - statePt.x) < radius && abs(center.y - statePt.y) < radius)
//            {
//                KF = createKalmanFilter(center.x, center.y);
//                std::cout << "new filter" << std::endl;
//            }
        }

//        if (avgMatches.size() > 0 && numPredictions < maxNumPredictions)
//        {
//                center.x = KF.statePre.at<float>(0);
//                center.y = KF.statePre.at<float>(1);
//                numPredictions++;
//        }



        // Either a match was found or we still want to make a prediction based on the previous prediction up to maxNumPredictions times
        if (avgMatches.size() > 0 || numPredictions < maxNumPredictions)
        {



            // First predict, to update the internal statePre variable
            cv::Mat prediction = KF.predict();
            cv::Point predictPt(prediction.at<float>(0),prediction.at<float>(1));

            cv::Mat_<float> measurement(2,1); measurement.setTo(cv::Scalar(0));
            measurement(0) = center.x;
            measurement(1) = center.y;

            // The "correct" phase that is going to use the predicted value and our measurement
            cv::Mat estimated = KF.correct(measurement);
            statePt = (cv::Point){(int)estimated.at<float>(0),(int)estimated.at<float>(1)};

            // draw blue cross at kalman filter estimation
            int l = 10; //radius of cross
            cv::line(img, (cv::Point){statePt.x-l,statePt.y}, (cv::Point){statePt.x+l,statePt.y}, cv::Scalar(255,0,0), 2);
            cv::line(img, (cv::Point){statePt.x,statePt.y-l}, (cv::Point){statePt.x,statePt.y+l}, cv::Scalar(255,0,0), 2);
        }


        // newImage is passed into the next filter
        cv::Mat newImage = cv::Mat(cv::Size(w,h), CV_8UC1, newDataPointer2);

        // Display images
//        cv::imshow("New Image", newImage);

//        cv::imshow("Binary Image", imgBin);

        cv::imshow("Original Image", img);

        // keep window open until any key is pressed
//        cv::waitKey(1);
        if(cv::waitKey(1) >= 0) break;
//        break;
    }
}

cv::KalmanFilter createKalmanFilter(int x, int y)
{
    // Create kalman filter
    cv::KalmanFilter KF(4, 2, 0);
    KF.transitionMatrix = *(cv::Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
//    cv::Mat_<float> measurement(2,1); measurement.setTo(cv::Scalar(0));

    // Initialize kalman filter
    KF.statePre.at<float>(0) = x;
    KF.statePre.at<float>(1) = y;
    KF.statePre.at<float>(2) = 0;
    KF.statePre.at<float>(3) = 0;
    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-4));  // lower values mean more prediction
    setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-3));  // lower values tighten on found points
    setIdentity(KF.errorCovPost, cv::Scalar::all(.1));

    return KF;
}
