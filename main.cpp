#include <QApplication>
#include <CL/cl.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>

#include "search.h"
#include "holdpoint.h"
#include "/opt/ros/groovy/include/opencv2/video/tracking.hpp"

// declare local methods
std::list<cv::Point> readMatches(search s, std::list<cv::Point> matches, int matchIndex, bool horz);
std::list<cv::Point> averageMatches(std::list<cv::Point> matches);
cv::Mat drawTargets(cv::Mat img, std::vector<HoldPoint> H, cv::Scalar color);
std::vector<HoldPoint> holdPoints(std::vector<HoldPoint> H, std::list<cv::Point> avgMatches);
cv::KalmanFilter createKalmanFilter(int x, int y);
cv::Point runKalmanFilter(cv::KalmanFilter KF, cv::Point statePt, std::list<cv::Point> avgMatches);

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
    search s3;
    search s4;
    search s5;
    search s6;
    cl_int win = 40;
    cl_double p1 = 0.4;
    cl_double p2 = 0.5;
    cl_double p3 = 0.6;
    s1.buildProgram(findSSLClPath, win, p1);
    s2.buildProgram(findSSLClPath, win, p1);
    s3.buildProgram(findSSLClPath, win, p2);
    s4.buildProgram(findSSLClPath, win, p2);
    s5.buildProgram(findSSLClPath, win, p3);
    s6.buildProgram(findSSLClPath, win, p3);

    // Create kalman filter
    cv::KalmanFilter KF = createKalmanFilter(0,0);
    cv::Point statePt;

    // Create vector of holdPoint filters for each marker
    std::vector<HoldPoint> H1;
    std::vector<HoldPoint> H2;
    std::vector<HoldPoint> H3;

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
            s3.setImage(imgBin);
            s4.setImage(imgBinVert);
            s5.setImage(imgBin);
            s6.setImage(imgBinVert);
            firstTime = false;
        }

        // Run OpenCV kernel to find markers
        s1.runProgram(imgBin);
        s2.runProgram(imgBinVert);
        s3.runProgram(imgBin);
        s4.runProgram(imgBinVert);
        s5.runProgram(imgBin);
        s6.runProgram(imgBinVert);
        // newDataPointer is used to display image
//        unsigned char* newDataPointer1 = (unsigned char*) s1.readOutput();
//        unsigned char* newDataPointer2 = (unsigned char*) s2.readOutput();
//        unsigned char* newDataPointer3 = (unsigned char*) s3.readOutput();
//        unsigned char* newDataPointer4 = (unsigned char*) s4.readOutput();
//        unsigned char* newDataPointer5 = (unsigned char*) s5.readOutput();
//        unsigned char* newDataPointer6 = (unsigned char*) s6.readOutput();
        int matchIndex1 = s1.readMatchesIndexOutput();
        int matchIndex2 = s2.readMatchesIndexOutput();
        int matchIndex3 = s3.readMatchesIndexOutput();
        int matchIndex4 = s4.readMatchesIndexOutput();
        int matchIndex5 = s5.readMatchesIndexOutput();
        int matchIndex6 = s6.readMatchesIndexOutput();


        // read matches from kernel
        std::list< cv::Point > matches1;
        std::list< cv::Point > matches2;
        std::list< cv::Point > matches3;
        matches1 = readMatches(s1, matches1, matchIndex1, true);
        matches1 = readMatches(s2, matches1, matchIndex2, false);
        matches2 = readMatches(s3, matches2, matchIndex3, true);
        matches2 = readMatches(s4, matches2, matchIndex4, false);
        matches3 = readMatches(s5, matches3, matchIndex5, true);
        matches3 = readMatches(s6, matches3, matchIndex6, false);

        // Average clusters
        std::list<cv::Point> avgMatches1 = averageMatches(matches1);
        std::list<cv::Point> avgMatches2 = averageMatches(matches2);
        std::list<cv::Point> avgMatches3 = averageMatches(matches3);

        H1 = holdPoints(H1, avgMatches1);
        H2 = holdPoints(H2, avgMatches2);
        H3 = holdPoints(H3, avgMatches3);

        // Draw targets over averaged matches
        img = drawTargets(img, H1, cv::Scalar(0,0,255));
        img = drawTargets(img, H2, cv::Scalar(0,255,255));
        img = drawTargets(img, H3, cv::Scalar(0,255,0));

        // run kalman filter
        /*
//        statePt = runKalmanFilter(KF, statePt, avgMatches1);

//        // draw blue cross at kalman filter estimation if there is a valid location
//        if (statePt != (cv::Point){-1,-1})
//        {
//            int l = 10; //radius of cross
//            cv::line(img, (cv::Point){statePt.x-l,statePt.y}, (cv::Point){statePt.x+l,statePt.y}, cv::Scalar(255,0,0), 2);
//            cv::line(img, (cv::Point){statePt.x,statePt.y-l}, (cv::Point){statePt.x,statePt.y+l}, cv::Scalar(255,0,0), 2);
//        }

//        // newImage is passed into the next filter
//        cv::Mat newImage = cv::Mat(cv::Size(w,h), CV_8UC1, newDataPointer1);
*/

        //        // Display images
//        cv::imshow("New Image", newImage);

//        cv::imshow("Binary Image", imgBin);

        cv::imshow("Original Image", img);

        // keep window open until any key is pressed
//        cv::waitKey(1);
        if(cv::waitKey(1) >= 0) break;
//        break;
    }
}



std::list<cv::Point> readMatches(search s, std::list<cv::Point> matches, int matchIndex, bool horz)
{
    // Read all matches from OpenCL kernel
    if (matchIndex > 0)
    {
        unsigned int* newMatchesPointer = s.readMatchesOutput(matchIndex);

        // loop through matches
        for (int i = 0; i < matchIndex; i++)
        {
            cv::Point match;
            // need to know if the kernel ran horizontally or vertically because x and y are flipped
            if (horz)
            {
                match = (cv::Point){newMatchesPointer[2*i], newMatchesPointer[2*i+1]};
            }
            else //vertical
            {
                match = (cv::Point){newMatchesPointer[2*i+1], newMatchesPointer[2*i]};
            }
            matches.push_front(match);

//                // Color a point at each match
//                cv::circle(img, matches.front(), 3, cv::Scalar(0,255,0), -1);
        }
    }
    return matches;
}

std::list<cv::Point> averageMatches(std::list<cv::Point> matches)
{
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
        int minClusterSize = 15;
        if (count > minClusterSize)
        {
            cv::Point avgMatch (xsum/count, ysum/count);
            avgMatches.push_front(avgMatch);
        }
    }
    return avgMatches;
}

cv::Mat drawTargets(cv::Mat img, std::vector<HoldPoint> H, cv::Scalar color)
{
    // Draw red taget over averaged matches
    for (std::vector<HoldPoint>::iterator it = H.begin(); it != H.end(); it++)
    {
        int l = 10; //radius of cross
        cv::Point center = it->heldMatch;

        cv::line(img, (cv::Point){center.x-l,center.y}, (cv::Point){center.x+l,center.y}, color, 2);
        cv::line(img, (cv::Point){center.x,center.y-l}, (cv::Point){center.x,center.y+l}, color, 2);
    }
    return img;
}

std::vector<HoldPoint> holdPoints(std::vector<HoldPoint> H, std::list<cv::Point> avgMatches)
{
    while (avgMatches.size() > 0)
    {
        bool matched = false;
        int radius = 30;
        // loops through all current matches
        for (std::vector<HoldPoint>::iterator it = H.begin(); it != H.end(); it++)
        {
            // update hold point if it is near a new match
            if (abs(avgMatches.front().x - it->prevPoint.x) < radius && abs(avgMatches.front().y - it->prevPoint.y) < radius)
            {
                it->update(avgMatches.front());
                matched = true;
                it->checked = true;
            }
        }

        // create new HoldPoint object if a avgMatch does not match any already existing
        if (!matched)
        {
            HoldPoint h;
            h.update(avgMatches.front());
            H.push_back(h);
        }
        avgMatches.pop_front();
    }

    for (std::vector<HoldPoint>::iterator it = H.begin(); it != H.end(); it++)
    {
        if (it->heldMatch == (cv::Point)NULL)
        {
            H.erase(it);
            break; // because iteration length has now changed
        }

        // calls update on all holdPoints that didn't match any new matches
        if (!it->checked)
        {
            it->update((cv::Point)NULL);
        }
        else // make sure all holdPoints start the next loop with checked as false
        {
            it->checked = false;
        }
    }
    return H;
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

cv::Point runKalmanFilter(cv::KalmanFilter KF, cv::Point statePt, std::list<cv::Point> avgMatches)
{
    // Run Kalman filter

    // numPredictions counts the number of frames since the last positive match
    static int numPredictions = 0;
    // maxNumPredictions is the number of frames the filter will guess a position since the last positive match
    static int maxNumPredictions = 5;

    // Get target location if a match was found
    cv::Point center;
    if (avgMatches.size() > 0)
    {
        int radius = 40;

        std::cout << "matches found" << std::endl;

        // loop through all average matches to see if any are close to previous match
        std::list< cv::Point > avgMatchesCopy = avgMatches;
        for (int i = 0; i < avgMatchesCopy.size(); i++)
        {
//                std::cout << "search" << std::endl;
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
            std::cout << statePt << std::endl;
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

//        // If a match was found draw it
//        int l = 10; //radius of cross
//        cv::line(img, (cv::Point){center.x-l,center.y}, (cv::Point){center.x+l,center.y}, cv::Scalar(0,0,255), 2);
//        cv::line(img, (cv::Point){center.x,center.y-l}, (cv::Point){center.x,center.y+l}, cv::Scalar(0,0,255), 2);
    }
    // No match found
    // Predict position based on last prediction, don't do this more than maxNumPredictions times
    else if (numPredictions < maxNumPredictions)
    {
        center = (cv::Point){(int)KF.statePre.at<float>(0),(int)KF.statePre.at<float>(1)};
        numPredictions++;
    }
    // marker position not known
    else
    {
        statePt =  (cv::Point){-1,-1};
    }

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
    }
    return statePt;
}
