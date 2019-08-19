#include "ReLULayer.h"



std::vector<Mat1f> ReLULayer::use(std::vector<Mat1f> in) {
    std::vector<Mat1f> out;
    for(Mat1f mat : in){
        eWMOp(mat,ReLU);
        out.push_back(mat.clone());
    }
    inputHistory = copyVec(out);
    return out;

}

std::vector<Mat1f> ReLULayer::dErr(std::vector<Mat1f> in) {
    std::vector<Mat1f> out = std::vector<Mat1f>(in.size());
    for(int m = 0;m<in.size();m++){

        eWMOp(inputHistory[m],dReLU);
        out[m] = in[m].mul(inputHistory[m]);

    }

    return out;
}

void ReLULayer::applyError() {
    //no error to apply here
    return;
}

ReLULayer::ReLULayer() {
    layerType = "ReLU";
    return;
}
