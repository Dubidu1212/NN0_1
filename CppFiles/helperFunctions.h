
#ifndef NN0_1_HELPERFUNCTIONS_H
#define NN0_1_HELPERFUNCTIONS_H

#include <opencv2/opencv.hpp>
#include <math.h>

//TODO: return pointers instead of objects
using namespace cv;

void addNumToVec(std::vector<Mat1f> in, float num);/*!<Adds a float to each element of each matrix in a vector*/
//static void eWVOp(std::vector<Mat1f> in, float (op)(float));/*!< Elementwise vector operation */
//TODO: check whether operator type has to be changed. to adapt for Mat3f
void eWMOp(Mat1f &in, float (op)(float)); /*!< Elementwise matrix operation */
std::vector<Mat1f> copyVec(std::vector<Mat1f> in);/*!<
 * Copies a Vector of Matrices.
 * @param in Vector to be copied
 * @return Copied Vector
 */

//!Copy dimensions of vector of matrices.
/*!
 * Creates a vector of matrices with the dimensions copied from an other vector
 * @param in Vector whose dimensions will get copied
 * @param fillVal value to be filled in
 * @return Copied Vector
 */
std::vector<Mat1f> copyDimVec(std::vector<Mat1f> in,float fillVal = 0);
float sigmoid(float in); /*!< Sigmoid activation function */
float sigmoidPrime(float in);/*!< Derivative of sigmoid function */
//!Rectified linear Unit: returns max(x,0)
float ReLU(float in);
//!Derivative of ReLU
float dReLU(float in);

//!Reverses an int. Copied from: http://eric-yuan.me/cpp-read-mnist/
int ReverseInt(int in);

//!Reads the Mnist database into a vec<mat>. Copied from: http://eric-yuan.me/cpp-read-mnist/
void read_Mnist(std::string filename,std::vector<Mat> &vec);

//!Reads the Mnist database labels into a vec<double>. Copied from: http://eric-yuan.me/cpp-read-mnist/
void read_Mnist_Label(std::string filename,std::vector<double> &vec);


#endif //NN0_1_HELPERFUNCTIONS_H
