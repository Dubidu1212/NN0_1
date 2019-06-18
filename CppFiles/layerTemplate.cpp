#include "layerTemplate.h"

layerTemplate::layerTemplate(int t, int fS, int nF, int s) {
    if(t != Convolutional){
        throw "Wrong layerTemplate Initializer: used initializer 'Convolutional'";
    }
    filterSize = fS;

    numFilters = nF;

    CStride = s;

}

layerTemplate::layerTemplate(int t) {
    if(t != ReLU){
        throw "Wrong layerTemplate Initializer: used initializer 'ReLU'";
    }

}

layerTemplate::layerTemplate(int t,int pS, int s) {
    if(t != Maxpool){
        throw "Wrong layerTemplate Initializer: used initializer 'Maxpool'";
    }
    poolingSize = pS;
    MaxStride = s;

}

layerTemplate::layerTemplate(int t, int s) {
    if(t != FullyConnected){
        throw "Wrong layerTemplate Initializer: used initializer 'Maxpool'";
    }
    size = s;

}
