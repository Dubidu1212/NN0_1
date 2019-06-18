#ifndef NN0_1_RELULAYER_H
#define NN0_1_RELULAYER_H

#include "NetworkLayer.h"

class ReLULayer : public NetworkLayer {
    //!Sets all values of the matrix smaller than 0 to 0 **Important! use cloned matrix**
    Mat1f use(Mat1f in);
    Mat1f dErr(Mat1f in);

};


#endif //NN0_1_RELULAYER_H
