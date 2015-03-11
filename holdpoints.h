#ifndef HOLDPOINTS_H
#define HOLDPOINTS_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <list>
#include <iostream>

class HoldPoints
{
public:
    HoldPoints();
    ~HoldPoints();
    void update(cv::Point avgMatches);

    int count;
    int timeout;
    bool checked;
    cv::Point prevPoint;
    cv::Point heldMatch;
};

#endif // HOLDPOINTS_H
