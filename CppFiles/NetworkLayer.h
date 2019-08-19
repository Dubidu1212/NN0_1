#ifndef NN0_1_NETWORKLAYER_H
#define NN0_1_NETWORKLAYER_H

#include "helperFunctions.h"


//! Superclass of all neural network layers
class NetworkLayer {
    //!value with which the training speed is determined
public:
    float lambda;
    //!Accumulates error until applied in applyError
    std::string layerType;

    //TODO: make general so it is possible to put in a Mat3f
    //! Takes as a input a Matrix and applies the layers operator on it
    virtual std::vector<Mat1f> use(std::vector<Mat1f> in) = 0;

    //! Gives d(err)/d(in) of this layer while calculating the error on the layer and putting it into error acumulate
    /*!
     *
     * @param in the error of the next layer
     * @return
     */
    virtual std::vector<Mat1f> dErr(std::vector<Mat1f> in) = 0;

    //! applies the saved errors. Should be used after the whole batch has accumulated dErr's
    virtual void applyError() = 0;


};


#endif //NN0_1_NETWORKLAYER_H
