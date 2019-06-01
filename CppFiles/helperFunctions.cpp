#include "helperFunctions.h"


void eWVOp(std::vector<Mat1f> in, float (*op)(float)) {

}

void eWMOp(Mat1f &in, float (*op)(float)) {
    for(int row = 0; row < in.rows; ++row) {
        for(int col = 0; col < in.cols; ++col) {
            in.at<float>(row,col) = op(in.at<float>(row,col));
        }
    }
}

float sigmoid(float in) {
    return 1/(1+exp(in*-1));
}

float sigmoidPrime(float in) {
    return sigmoid(in)*(1-sigmoid(in));
}

std::vector<Mat1f> copyVec(std::vector<Mat1f> in) {
    std::cerr<< "not yet implemented" << std::endl;
    return std::vector<Mat1f>();
}



