
#ifndef NN0_1_HELPERFUNCTIONS_H
#define NN0_1_HELPERFUNCTIONS_H

#include <opencv2/opencv.hpp>
#include <math.h>
using namespace cv;

static void eWVOp(std::vector<Mat1f> in, float (op)(float));/*!< Elementwise vector operation */
//TODO: check whether operator type has to be changed.
static void eWMOp(Mat1f &in, float (op)(float)); /*!< Elementwise matrix operation */
static std::vector<Mat1f> copyVec(std::vector<Mat1f> in);/*!<
 * Copies a Vector of Matrices.
 * @param in Vector to be copied
 * @return Copied Vector
 */
static float sigmoid(float in); /*!< Sigmoid activation function */
static float sigmoidPrime(float in);/*!< Derivative of sigmoid function */


#endif //NN0_1_HELPERFUNCTIONS_H
