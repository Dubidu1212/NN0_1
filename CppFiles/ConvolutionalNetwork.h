#ifndef NN0_1_CONVOLUTIONALNETWORK_H
#define NN0_1_CONVOLUTIONALNETWORK_H

#include "helperFunctions.h"
#include "layerTemplate.h"
using namespace cv;



class ConvolutionalNetwork {
public:

    /*!
     *
     * @param layerSetup
     * @param l
     */
    ConvolutionalNetwork(std::vector<layerTemplate> layerSetup, float l);

    Mat1f use(Mat in);

private:


};


#endif //NN0_1_CONVOLUTIONALNETWORK_H
