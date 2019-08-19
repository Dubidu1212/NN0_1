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
    layerTemplate = layersizes;
    lambda = l;
    for(int layer = 0;layer<layers;layer++){
        Mat1f tempMat(layersizes[layer],1);
        randu(tempMat,-1.0,1.0);
        b.push_back(tempMat.clone());
    }
    for(int layer = 0;layer<layers-1;layer++){
        Mat1f tempMat(layersizes[layer+1],layersizes[layer]);
        randu(tempMat,-1.0,1.0);
        w.push_back(tempMat.clone());
    }

}

void FeedForwardNetwork::save(std::string filename) {
    std::ofstream savefile;
    savefile.open(filename);
    //layersizes
    savefile << layerTemplate.size() << "\n";
    for(int ls =0;ls<layerTemplate.size()-1;ls++){
        savefile << layerTemplate[ls] << " ";
    }
    savefile << layerTemplate[layerTemplate.size()-1] << "\n";

    //biases
    savefile << "b \n";
    savefile << b.size() << "\n";
    for(int bias = 0;bias<b.size();bias++){
        savefile << b[bias].rows << "\n";
        for(int val = 0;val< b[bias].rows-1;val++){
            savefile << b[bias].at<float>(val,0) << " ";
        }
        savefile << b[bias].at<float>(b[bias].rows-1,0) << "\n" ;
    }

    //weights
    savefile << "w \n";
    savefile << w.size() << "\n";
    for(int weight = 0;weight<w.size();weight++){
        savefile << w[weight].rows << " " << w[weight].cols << "\n";

        for(int r = 0;r< w[weight].rows;r++){

            for(int c = 0;c<w[weight].cols;c++){
                savefile << w[weight].at<float>(r,c) << " ";
            }
            savefile << "\n";
        }

    }
    savefile << "l\n";
    savefile << lambda;
    std::cout << "saved file: "<< filename <<"\n";

}

FeedForwardNetwork::FeedForwardNetwork(std::string filename){

    std::ifstream in;
    in.open(filename);
    assert(in.is_open());
    //layersizes
    int numL;
    in >> numL;
    std::vector<int> ls;
    for(int x = 0;x<numL;x++){
        int temp;
        in >> temp;
        ls.push_back(temp);
    }
    layers = ls.size();
    layerTemplate = ls;

    //biases
    char test;
    in >> test;
    assert(test == 'b');


    //biases
    int bs;
    in >> bs;
    for(int x = 0;x<bs;x++){
        int biass;
        in >> biass;
        Mat1f bmat = Mat1f(biass,1);
        for(int y = 0;y<biass;y++){
            float biasval;
            in >> biasval;
            bmat.at<float>(y,0) = biasval;
        }
        b.push_back(bmat.clone());
    }


    //weights

    in >> test;
    assert(test == 'w');

    int ws;
    in >> ws;
    for(int x = 0;x<ws;x++){
        int rs,cs;
        in >> rs >> cs;
        Mat1f currW = Mat1f(rs,cs);
        for(int r = 0;r<rs;r++){
            for(int c = 0;c<cs;c++){
                float wval;
                in >> wval;
                currW.at<float>(r,c) = wval;
            }
        }
        w.push_back(currW);

    }

    //lambda
    in >> test;
    assert(test == 'l');
    in >> lambda;

    std::cout << "loaded file: " << filename <<  "\n";
}

