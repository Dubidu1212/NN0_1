#ifndef NN0_1_FULLYCONNECTEDLAYER_H
#define NN0_1_FULLYCONNECTEDLAYER_H

#include "NetworkLayer.h"

class FullyConnectedLayer : public NetworkLayer{
public:
    //!The number of nodes in this layer.
    int nodes;

    //!number of times the error has not been applied
    int passes = 0;

    std::string activationFunction = "Sigmoid";

    Mat1f weightsError;
    Mat1f weights;
    Mat1f biasesError;
    Mat1f biases;
    //!last value before sigmoid was applied
    Mat1f lastpreSig;

    //!last softmax. Created while passing forward through a softmax layer
    Mat1f lastSoftmax;

    //!last value of n eg output of the preceding layer
    Mat1f nodeHistory;

    //!has the size of the input to be copied when propagating the error from this to the last layer
    std::vector<Mat1f> inputDimensions;

    /*!
     *
     * @param in a std::vector of matricies
     * @return a std::vector containing one matrix at position 0
     */
    std::vector<Mat1f> use(std::vector<Mat1f> in) override ;
    std::vector<Mat1f> dErr(std::vector<Mat1f> in) override;
    FullyConnectedLayer(int nodes,int inputSize,float lambda,std::string activationFunction);
    std::tuple<int,int,int> outputSize(std::tuple<int,int,int> in);

    void applyError() override;
};


#endif //NN0_1_FULLYCONNECTEDLAYER_H
