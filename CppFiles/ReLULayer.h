#ifndef NN0_1_RELULAYER_H
#define NN0_1_RELULAYER_H

#include "NetworkLayer.h"

class ReLULayer : public NetworkLayer {
    //!Sets all values of the matrix smaller than 0 to 0 **Important! use cloned matrix**

    std::vector<Mat1f> use(std::vector<Mat1f> in);
    std::vector<Mat1f> dErr(std::vector<Mat1f> in);
    void applyError();

    std::tuple<int,int,int> outputSize(std::tuple<int,int,int> in);
    std::vector<Mat1f> inputHistory;

public:
    ReLULayer();
};


#endif //NN0_1_RELULAYER_H
