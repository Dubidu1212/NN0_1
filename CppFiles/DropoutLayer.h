#ifndef NN0_1_DROPOUTLAYER_H
#define NN0_1_DROPOUTLAYER_H


#include "NetworkLayer.h"
#include <random>

class DropoutLayer : public NetworkLayer{
public:

    //!Defines how often the DropoutMats are reloaded
    /*!
     * DropoutMat is reloaded every reloadPeriod time the network is passed
     */
    int reloadPeriod;

    //!number of times the layer has been passed during training
    int passes = 0;

    //!contains the matrices where random positions are set to 0
    std::vector<Mat1f> DropoutMats;

    //!Indicates whether network is training or using the layer
    bool training = true;

    //TODO: set up in constructor
    std::uniform_int_distribution<int> distr;
    std::uniform_int_distribution<int> distc;


    std::mt19937 mt;


    //!Percentage of Neurons randomly set to 0
    float dropoutPercentage;

    std::vector<Mat1f> use(std::vector<Mat1f> in);

    std::vector<Mat1f> dErr(std::vector<Mat1f> in);

    void applyError() override;

    std::tuple<int,int,int> outputSize(std::tuple<int,int,int> in);

    //!input dimensions
    std::tuple<int,int,int> inputDim;

    /*!
     *
     * @param relTime how often is the zeroMat refreshed
     * @param dropoutP how many percent of the data is set to 0
     * @param inputSize how is the input formed rows,cols,parallels
     */
    DropoutLayer(int relTime,float dropoutP,std::tuple<int,int,int> inputSize);
};


#endif //NN0_1_DROPOUTLAYER_H
