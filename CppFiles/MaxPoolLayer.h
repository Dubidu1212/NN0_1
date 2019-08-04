#ifndef NN0_1_MAXPOOLLAYER_H
#define NN0_1_MAXPOOLLAYER_H

#include "NetworkLayer.h"

class MaxPoolLayer :public NetworkLayer{
private:
    int s;
    int poolSize;

public:
    MaxPoolLayer(int poolingSize, int stride);
    std::vector<Mat1f> use(std::vector<Mat1f> in);

};


#endif //NN0_1_MAXPOOLLAYER_H
