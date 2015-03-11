#ifndef HOLDPOINT_H
#define HOLDPOINT_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <list>
#include <iostream>

class HoldPoint
{
public:
    HoldPoint();
    ~HoldPoint();
    void update(cv::Point avgMatches);

    int count;
    int timeout;
    bool checked;
    cv::Point prevPoint;
    cv::Point heldMatch;
};

#endif // HOLDPOINT_H
