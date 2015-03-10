#ifndef HOLDPOINTS_H
#define HOLDPOINTS_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <list>

class HoldPoints
{
public:
    HoldPoints();
    void update(std::list<cv::Point> avgMatches);

    int count;
    int timeout;
    cv::Point prevPoint;
    std::list <cv::Point> heldMatches;
};

#endif // HOLDPOINTS_H
