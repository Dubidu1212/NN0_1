#ifndef NN0_1_NETWORKLAYER_H
#define NN0_1_NETWORKLAYER_H

#include "helperFunctions.h"


//! Superclass of all neural network layers
class NetworkLayer {//TODO: subclass this to make different layers like convolutional and relu etc.
    //TODO: make general so it is possible to put in a Mat3f
    //! Takes as a input a Matrix and applies the layers operator on it
    virtual std::vector<Mat1f> use(std::vector<Mat1f> in) = 0;

    //! Gives d(out)/d(in) of this layer
    virtual std::vector<Mat1f> dErr(std::vector<Mat1f> in) = 0;

};


#endif //NN0_1_NETWORKLAYER_H
