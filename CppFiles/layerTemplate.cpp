#include "layerTemplate.h"

layerTemplate::layerTemplate(int t, int fS, int nF, int s) {
    if(t != ConvolutionalLayer){
        throw "Wrong layerTemplate Initializer: used initializer 'Convolutional'";
    }
    filterSize = fS;

    numFilters = nF;

    CStride = s;

}

layerTemplate::layerTemplate(int t) {
    if(t != ReLULayer){
        throw "Wrong layerTemplate Initializer: used initializer 'ReLU'";
    }

}

layerTemplate::layerTemplate(int t,int pS, int s) {
    if(t != MaxpoolLayer){
        throw "Wrong layerTemplate Initializer: used initializer 'Maxpool'";
    }
    poolingSize = pS;
    MaxStride = s;

}

layerTemplate::layerTemplate(int t, int s) {
    if(t != FullyConnectedLayer){
        throw "Wrong layerTemplate Initializer: used initializer 'Maxpool'";
    }
    size = s;

}
