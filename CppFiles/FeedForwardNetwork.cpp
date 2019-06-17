#include "FeedForwardNetwork.h"



Mat1f FeedForwardNetwork::use(Mat1f in){
    for(int layer = 0;layer < layers-1;layer++){
        in+=b[layer];
        in = w[layer]*in;

        eWMOp(in,sigmoid);

    }

    return in;//maybe .clone()
}

void FeedForwardNetwork::trainBatch(std::vector<std::pair<Mat1f, Mat1f>> batch) {

}


void FeedForwardNetwork::trainSingle(Mat1f in, Mat1f out) {
    std::vector<Mat1f> nodeHistory = copyVec(b);//!<N
    std::vector<Mat1f> praesigHistory = copyVec(b);
    nodeHistory[0] = in;
    praesigHistory[0] = in;

    //forward propagation
    for(int layer = 0;layer<layers-1;layer++){
        nodeHistory[layer+1] = w[layer]*(nodeHistory[layer]+b[layer]);//maybe copy
        praesigHistory[layer+1] = w[layer]*(nodeHistory[layer]+b[layer]);
        eWMOp(nodeHistory[layer+1],sigmoid);
    }

    //backpropagation

    //initializing error matrices
    std::vector<Mat1f> Ew = copyDimVec(w);
    std::vector<Mat1f> En = copyDimVec(nodeHistory);
    std::vector<Mat1f> Eb = copyDimVec(b);


    for(int layer = layers-1;layer>=0;layer--){
        if(layer == layers-1){
            //Last layer to Error
            En[layer] = 2*(nodeHistory[layer]-out);// d(Error)/d(postsig)
        }
        else{
            Mat1f dsig;//Vector d(postisg)/d(präsig) == d(nodeHistory[i])/d(präsigHistory[i])
            dsig = praesigHistory[layer+1].clone();//maybe only layer
            eWMOp(dsig,sigmoidPrime);

            Mat1f dErr = En[layer+1].mul(dsig);// d(Error)/d(Postsig) * d(postsig)/d(präsig) = d(Error)/d(Präsig)



            En[layer] = w[layer].t() * dErr;//maybe clone
            Eb[layer] = w[layer].t() * dErr;
            Ew[layer] = dErr* (b[layer]+nodeHistory[layer]).t();




        }

    }

    //Move in direction of negative gradient

    for(int layer = 0;layer < layers;layer++){
        b[layer]-=(Eb[layer]*lambda);
        if(layer < layers-1){
            w[layer]-=(Ew[layer]*lambda);
        }

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

FeedForwardNetwork::FeedForwardNetwork(std::vector<int> layersizes, float l) {
    layers = layersizes.size();
    lambda = l;
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

