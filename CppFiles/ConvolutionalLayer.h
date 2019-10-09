#ifndef NN0_1_CONVOLUTIONALLAYER_H
#define NN0_1_CONVOLUTIONALLAYER_H

#include "NetworkLayer.h"

class ConvolutionalLayer:public NetworkLayer {

public:



    //!number of times the error hasnt been applied
    int passes = 0;

    //!finds d(out)/d(in) of a single mat
    /*!
     *
     * @param outMat position in the output vector of the matrix
     * @return returns d(out)/d(in)
     */
    Mat1f ErrdInSingle(int outMat);

    //!Returns d(out)/d(in) for the whole layer
    std::vector<Mat1f> dErr(std::vector<Mat1f> in);//TODO: make general by naming functions the same as in NetowrkLayer.h

    //!computes and saves d(out)/d(err)

    void ErrdFilter();//TODO: combine with outdin

    //!applies the error accumulated after a batch
    void applyError();

    //! applies convolution to this layer using the opencv method which is fast due to fast fourier
    std::vector<Mat1f> use(std::vector<Mat1f> in);

    //! applies convolution on this layer using my slow method due to no fast fourier :-(
    std::vector<Mat1f> ownUse(std::vector<Mat1f> in);

    //!size of the filters
    int filterSize;
    std::vector<Mat1f> filters;

    //!Maps a given output matrix to a filter and a input matrix
    /*!
     *
     * At position i of OutInMapping there is a pair of ints.
     *
     * The first int indicates the position of the input matrix in the input vector
     *
     * The second int indicates the filter used.
     *
     * Example:
     * input = vec<mat>
     * output = vec<mat>
     * output = use(input)
     *
     * filter used on output[i] = OutInMapping[i].second
     * beginning of output[i] = OutInMapping[i].first
     */
    std::vector<std::pair<int,int>> OutInMapping;//TODO: remove to save performance

    //!The last input
    std::vector<Mat1f> inputHistory;

    //! error of the next layer
    std::vector<Mat1f> nextLayerErr;//TODO: initialize. Has to be set by the class containing all the layers

    //!here the errors get summed up until they get applied in applyError()
    std::vector<Mat1f> errorAccumulate;

    //!The Rows of the input processed in this layer. Is initialized only after usage
    int inputDimensionsRows;

    //!Number of Cols of the input processed in this layer. Is initialized only after usage of layer.
    int inputDimensionsCols;

    //!Calculates the output size given the output
    std::tuple<int,int,int> outputSize(std::tuple<int,int,int> in);
public:
    //!Initializer for Convolutional Layer
    /*!
     *
     * @param filterSize Size of the filters in this layer
     * @param numFilters Number of filters in this layer
     * @param lambda learning rate
     */
    ConvolutionalLayer(int filterSize,int numFilters,float lambda);
};


#endif //NN0_1_CONVOLUTIONALLAYER_H
