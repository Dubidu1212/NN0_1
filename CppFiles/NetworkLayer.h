#ifndef NN0_1_NETWORKLAYER_H
#define NN0_1_NETWORKLAYER_H

#include "helperFunctions.h"


//! Superclass of all neural network layers
class NetworkLayer {//TODO: subclass this to make different layers like convolutional and relu etc.
    //TODO: make general so it is possible to put in a Mat3f
    //! Takes as a input a Matrix and applies the layers operator on it
    virtual Mat1f use(Mat1f in) = 0;

    //! Gives d(out)/d(in) of this layer
    virtual Mat1f dErr(Mat1f in) = 0;

};


#endif //NN0_1_NETWORKLAYER_H
