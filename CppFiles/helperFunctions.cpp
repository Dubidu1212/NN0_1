#include "helperFunctions.h"


void addNumToVec(std::vector<Mat1f> in, float num){
    for(Mat1f m : in){
        for(int row = 0; row < m.rows; ++row) {
            for(int col = 0; col < m.cols; ++col) {
                m.at<float>(row,col) += num;
            }
        }
    }
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
    //TODO: make pointer
    std::vector<Mat1f> retVec;
    for(int x = 0;x<in.size();x++){
        retVec.push_back(in[x].clone());
    }

    return retVec;
}

std::vector<Mat1f> copyDimVec(std::vector<Mat1f> in,float fillVal) {
    std::vector<Mat1f> retVec;
    for(int i = 0;i<in.size();i++){
        retVec.emplace_back(Mat1f(in[i].rows,in[i].cols,fillVal));//maybe push_back
    }
    return retVec;
}


