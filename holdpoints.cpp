#include "holdpoints.h"

HoldPoints::HoldPoints()
{
    count = 0;
    timeout = 10;
    checked = false;
}

void HoldPoints::update(cv::Point avgMatch)
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
    else
    {
        std::cout << "debug2" <<  std::endl;
    }
}

HoldPoints::~HoldPoints()
{
}

