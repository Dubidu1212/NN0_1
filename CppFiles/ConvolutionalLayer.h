#ifndef NN0_1_CONVOLUTIONALLAYER_H
#define NN0_1_CONVOLUTIONALLAYER_H

#include "NetworkLayer.h"

class ConvolutionalLayer:public NetworkLayer {
    Mat1f use(Mat1f in);
    int filterSize;
    int numFilters;
    int stride;
    std::vector<Mat1f> filters;

};


#endif //NN0_1_CONVOLUTIONALLAYER_H
