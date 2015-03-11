#include "holdpoint.h"

HoldPoint::HoldPoint()
{
    count = 0;
    timeout = 10;
    checked = false;
}

void HoldPoint::update(cv::Point avgMatch)
{
    heldMatch = (cv::Point)NULL;
    if (avgMatch != (cv::Point)NULL)
    {
        prevPoint = avgMatch;
        heldMatch = avgMatch;
        count = 0;
    }
    else if (count < timeout)
    {
        count++;
        heldMatch = prevPoint;
    }
}

HoldPoint::~HoldPoint()
{
}

