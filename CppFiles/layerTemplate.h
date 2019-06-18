//
// Created by raphael on 18.06.19.
//

#ifndef NN0_1_LAYERTEMPLATE_H
#define NN0_1_LAYERTEMPLATE_H

#include <string>

enum layerType {Convolutional,ReLU, Maxpool, FullyConnected};
struct layerTemplate{


    //!Type of the Layer. Can be Convolutional, ReLU,Maxpool or FulllyConnected;
    int Type;

    //Convolutional
    //!Initializer for layerTemplate of type Convolutional
    /*!
     * The Convolutional layer is the main part of the convolutional network.
     * It works by convolving (shifting over the whole 2d matrix) a smaller matrix, making dot products of the two and saving them in another matrix
     * The resulting layer has the dimensions *inputlayersize - filtersize + 1*
     * @param t type, has to be Convolutional
     * @param fS Size of the filter, produces a fS x fS filter
     * @param nF Number of filters in this layer
     * @param s Stride of the filter, probably unused
     */
    layerTemplate(int t, int fS, int nF, int s);
    //!Size of the filters
    int filterSize;
    //!Number of filters
    int numFilters;
    //!Stride of the filter
    int CStride;



    //ReLU
    //!Initializer for layerTemplate of type ReLU
    /*!
     *  Rectified Linear Unit is the activation function of the convolutional network. It sets all negative values to 0
     *  The resulting layer has the same size as the input layer
     * @param t type, has to be ReLU
     */
    layerTemplate(int t);



    //Maxpool
    //!Initializer of layerTemplate of type Maxpool
    /*!
     *  Maxpool is used to reduce the complexity of the network.
     *  It works by taking a part of size pS of the matrix and convolving (moving) it over the matrix it makes steps of size s.
     *  At each point it saves the maximum of this window.
     *  The size of the resulting layer is *ceil(layersize/stride)
     * @param t type, has to be Maxpool
     * @param pS size of the pool
     * @param s Stride to be used in this pooling layer
     */
    layerTemplate(int t, int pS,int s);
    //! Size of the pool
    int poolingSize;
    //! Stride in this maxpool layer
    int MaxStride;



    //FullyConnected
    //!Initializer of layerTemplate of type FullyConnected
    /*!
     * Normal feedforward layer used in a normal multilayer perceptron.
     * @param t type, has to be FullyConnected
     * @param size size of the layer
     */
    layerTemplate(int t, int s);
    int size;

};



#endif //NN0_1_LAYERTEMPLATE_H
