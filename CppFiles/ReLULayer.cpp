#include "ReLULayer.h"


std::vector<Mat1f> ReLULayer::use(std::vector<Mat1f> in) {
    std::vector<Mat1f> out;
    for(Mat1f mat : in){
        eWMOp(mat,ReLU);
        out.push_back(mat.clone());
    }
    return out;

}

std::vector<Mat1f> ReLULayer::dErr(std::vector<Mat1f> in) {
    std::vector<Mat1f> out;
    for(Mat1f mat : in){
        eWMOp(mat,dReLU);
        out.push_back(mat.clone());
    }
    return out;
}

void ReLULayer::applyError() {
    //no error to apply here
    return;
}
