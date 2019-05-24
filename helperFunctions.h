
#ifndef NN0_1_HELPERFUNCTIONS_H
#define NN0_1_HELPERFUNCTIONS_H

#include <opencv2/opencv.hpp>
using namespace cv;

static void eWVOp(std::vector<Mat1f> in, float (op)(float));//TODO: check whether operator type has to be changed.
static void eWMOp(Mat1f &in, float (op)(float));
static float sigmoid(float in);
static float sigmoidPrime(float in);


#endif //NN0_1_HELPERFUNCTIONS_H
