#include "holdpoints.h"

HoldPoints::HoldPoints()
{
    count = 0;
    timeout = 10;
}

void HoldPoints::update(std::list<cv::Point> avgMatches)
{
    heldMatches.clear();
    if (avgMatches.size() > 0)
    {
        prevPoint = avgMatches.front();
        heldMatches.push_front(avgMatches.front());
        count = 0;
    }
    else if (count < timeout)
    {
        count++;
        heldMatches.push_front(prevPoint);
    }
}

