#ifndef NN0_1_CONVOLUTIONALNETWORK_H
#define NN0_1_CONVOLUTIONALNETWORK_H

#include "helperFunctions.h"
#include "NetworkLayer.h"

using namespace cv;



class ConvolutionalNetwork {

    int input_width,input_heigth,num_classes;
    long long propagations = 0;
    float lambda;
    std::string lossFunction;
public:

    std::vector<NetworkLayer*> layers;

    //!propagates the matrix in trough the network
    Mat1f use(Mat1f in);

    //!propagate the error backwards through the network without applying it
    /*!
     *
     * @param error the error of the network in the last layer
     */
    void dErr(Mat1f error);

    //!propagates forwards and backwards
    void wholePropagation(Mat1f in, Mat1f desiredOutput);

    //!applies the error which has been accumulated through dErr
    /*!
     * should be called after each batch
     */
    void applyError();

    void save(std::string filename);

    //!loads a network from a file
    ConvolutionalNetwork(std::string filename);

    ConvolutionalNetwork(int input_width,int input_height,int num_classes, float lambda, std::string lossFunction);


};


#endif //NN0_1_CONVOLUTIONALNETWORK_H
