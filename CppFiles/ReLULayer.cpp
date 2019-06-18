#include "ReLULayer.h"

Mat1f ReLULayer::use(Mat1f in) {
    eWMOp(in,ReLU);
    return in;
}

Mat1f ReLULayer::dErr(Mat1f in) {
    eWMOp(in,dReLU);
    return in;
}


