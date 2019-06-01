#include "FeedForwardNetwork.h"



Mat1f FeedForwardNetwork::use(Mat1f in){
    for(int layer = 0;layer < layers-1;layer++){
        in+=b[layer];
        in*=w[layer];
        eWMOp(in,sigmoid);

    }
    in+=b[layers-1];//maybe remove
    eWMOp(in,sigmoid);
    return in;//maybe .clone()
}

void FeedForwardNetwork::trainBatch(std::vector<std::pair<Mat1f, Mat1f>> batch) {

}


void FeedForwardNetwork::trainSingle(Mat1f in, Mat1f out) {
    std::vector<Mat1f> nodeHistory = copyVec(b);
    nodeHistory[0] = in;
    for(int layer = 0;layer<layers-2;layer++){//maybe -1
        nodeHistory[layer+1] = w[layer]*(nodeHistory[layer]+b[layer]);//maybe copy
    }


}




void FeedForwardNetwork::print() {
    std::cout << "==============Print================" << std::endl;
    std::cout << "Biases" << std::endl;
    for(int x = 0;x<b.size();x++){
        std::cout << "layer: " << x << std::endl;
        std::cout << b[x] << std::endl;
    }
    std::cout << "Weights" << std::endl;
    for(int x = 0;x<w.size();x++){
        std::cout << "layer: " << x << " to: " << x+1 << std::endl;
        std::cout << w[x] << std::endl;
    }

}

FeedForwardNetwork::FeedForwardNetwork(std::vector<int> layersizes) {
    layers = layersizes.size();
    for(int layer = 0;layer<layers;layer++){
        Mat1f tempMat(layersizes[layer],1);
        randu(tempMat,0.0,1.0);
        b.push_back(tempMat.clone());
    }
    for(int layer = 0;layer<layers-1;layer++){
        Mat1f tempMat(layersizes[layer+1],layersizes[layer]);
        randu(tempMat,0.0,1.0);
        w.push_back(tempMat.clone());
    }

}

