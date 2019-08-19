#ifndef NN0_1_MAXPOOLLAYER_H
#define NN0_1_MAXPOOLLAYER_H

#include "NetworkLayer.h"

class MaxPoolLayer :public NetworkLayer{
public:
    //!stride of the pool
    int s;
    //!size of the pool
    int poolSize;

    //!The Rows of the input processed in this layer. Is initialized only after usage
    int inputDimensionsRows;

    //!Number of Cols of the input processed in this layer. Is initialized only after usage of layer.
    int inputDimensionsCols;

    //!The last input
    std::vector<Mat1f> inputHistory;

    //!Initializes a MaxPooling layer
    /*!
     *
     * @param poolingSize Defines the size of the pool from which the maximum is taken
     * @param stride Defines how fast the Pool moves over the image
     */
    MaxPoolLayer(int poolingSize, int stride);
    std::vector<Mat1f> use(std::vector<Mat1f> in);
    std::vector<Mat1f> dErr(std::vector<Mat1f> in);
    void applyError();
};


#endif //NN0_1_MAXPOOLLAYER_H
