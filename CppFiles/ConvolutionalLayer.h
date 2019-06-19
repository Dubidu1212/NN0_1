#ifndef NN0_1_CONVOLUTIONALLAYER_H
#define NN0_1_CONVOLUTIONALLAYER_H

#include "NetworkLayer.h"

class ConvolutionalLayer:public NetworkLayer {

    //!Initializer for Convolutional Layer
    /*!
     *
     * @param filterSize Size of the filters in this layer
     * @param numFilters Number of filters in this layer
     */
    ConvolutionalLayer(int filterSize,int numFilters);

    std::vector<Mat1f> dErr(std::vector<Mat1f> in);

    //!finds d(out)/d(in) of a single mat
    /*!
     *
     * @param outMat position in the output vector of the matrix
     * @return returns d(out)/d(in)
     */
    Mat1f IndOutSingle(int outMat);

    Mat1f FilterdOut(int outMat);

    //! applies convolution to this layer using the opencv method which is fast due to fast fourier
    std::vector<Mat1f> use(std::vector<Mat1f> in);

    //! applies convolution on this layer using my slow method due to no fast fourier :-(
    std::vector<Mat1f> ownUse(std::vector<Mat1f> in);

    //!size of the filters
    int filterSize;
    std::vector<Mat1f> filters;

    //!Maps a given output matrix to a filter and a input matrix
    /*!
     * at position i of OutInMapping there is a pair of ints.
     * The first int indicates the position of the input matrix in the input vector
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
    std::vector<std::pair<int,int>> OutInMapping;
    std::vector<Mat1f> inputHistory;
    std::vector<Mat1f> nextLayerErr;

};


#endif //NN0_1_CONVOLUTIONALLAYER_H
