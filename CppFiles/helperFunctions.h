
#ifndef NN0_1_HELPERFUNCTIONS_H
#define NN0_1_HELPERFUNCTIONS_H

#include <opencv2/opencv.hpp>
using namespace cv;

static void eWVOp(std::vector<Mat1f> in, float (op)(float));/*!< Elementwise vector operation */
//TODO: check whether operator type has to be changed.
static void eWMOp(Mat1f &in, float (op)(float)); /*!< Elementwise matrix operation */
static float sigmoid(float in); /*!< Sigmoid activation function */
static float sigmoidPrime(float in);/*!< Derivative of sigmoid function */


#endif //NN0_1_HELPERFUNCTIONS_H
