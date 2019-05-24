#ifndef NN0_1_FEEDFORWARDNETWORK_H
#define NN0_1_FEEDFORWARDNETWORK_H

#include <opencv2/opencv.hpp>
#include "helperFunctions.h"

using namespace cv;

class FeedForwardNetwork{
private:
    std::vector<Mat1f> b;
    std::vector<Mat1f> w;
public:
    Mat1f use(Mat1f in);
    void trainBatch(std::vector<std::pair<Mat1f,Mat1f>> batch);
    void print();

};

#endif //NN0_1_FEEDFORWARDNETWORK_H
