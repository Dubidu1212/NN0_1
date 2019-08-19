#include "ConvolutionalNetwork.h"
#include "MaxPoolLayer.h"
#include "ReLULayer.h"
#include "ConvolutionalLayer.h"
#include "FullyConnectedLayer.h"

Mat1f ConvolutionalNetwork::use(Mat1f in) {
    std::vector<Mat1f> runtimeMatVec = std::vector<Mat1f>(1);
    runtimeMatVec[0] = in.clone();
    for(int l = 0;l<layers.size();l++){
        runtimeMatVec = layers[l]->use(runtimeMatVec);

    }
    return runtimeMatVec[0];//only works if last layer is fully connected
}

void ConvolutionalNetwork::dErr(Mat1f error) {
    std::vector<Mat1f> runtimeError = std::vector<Mat1f>(0);
    runtimeError.push_back(error.clone());
    for(int l = layers.size()-1;l>=0;l--){
        runtimeError = layers[l]->dErr(runtimeError);
    }
}

void ConvolutionalNetwork::applyError() {
    for(int l = 0;l<layers.size();l++){
        layers[l]->applyError();
    }
}

ConvolutionalNetwork::ConvolutionalNetwork(int input_width, int input_height, int num_classes, float lambda, std::string lossFunction) {
    this->lossFunction = lossFunction;
    this->input_width = input_width;
    this->input_heigth = input_height;
    this->num_classes = num_classes;
    this->lambda = lambda;
}

void ConvolutionalNetwork::wholePropagation(Mat1f in, Mat1f desiredOutput) {
    propagations++;
    if(lossFunction == "MSE"){
        Mat1f Error = (use(in)-desiredOutput)*2;
        dErr(Error);
    }
    else if(lossFunction == "CrossEntropy"){
        Mat1f Error = (use(in)-desiredOutput);
        dErr(Error);
    }
    else{
        std::cerr << "Fatal error while applying loss function: Function <" << lossFunction << "> doesn't exist" << std::endl;
    }

}

void ConvolutionalNetwork::save(std::string filename) {
    std::ofstream sf(filename);

    sf << input_width << std::endl;
    sf << input_heigth << std::endl;
    sf << num_classes << std::endl;
    sf << lambda << std::endl;
    sf << propagations << std::endl;
    sf << layers.size() << std::endl;
    for(int l = 0;l<layers.size();l++){
        std::string layerType = layers[l]->layerType;
        sf <<  layerType<< std::endl;
        if(layerType == "MaxPool"){
            MaxPoolLayer* layer = static_cast<MaxPoolLayer*>(layers[l]);
            sf << layer->s << std::endl;
            sf << layer->poolSize << std::endl;
        }
        else if(layerType == "ReLU"){

        }
        else if(layerType == "FullyConnected"){
            FullyConnectedLayer* layer = static_cast<FullyConnectedLayer*>(layers[l]);

            sf << layer->activationFunction << std::endl;

            sf << layer->nodes << std::endl;


            //biases
            sf << layer->biases.rows << "\n";
            for(int val = 0;val< layer->biases.rows-1;val++){
                sf <<layer->biases.at<float>(val,0) << " ";
            }
            sf << layer->biases.at<float>(layer->biases.rows-1,0) << "\n" ;

            //weights
            sf << layer->weights.rows << " " << layer->weights.cols << "\n";
            for(int r = 0;r< layer->weights.rows;r++){

                for(int c = 0;c<layer->weights.cols;c++){
                    sf << layer->weights.at<float>(r,c) << " ";
                }
                sf << "\n";
            }



        }
        else if(layerType == "Convolutional"){
            ConvolutionalLayer* layer = static_cast<ConvolutionalLayer*>(layers[l]);



            sf << layer->filterSize  << std::endl;
            sf << layer->filters.size() << std::endl;
            for(int f = 0;f<layer->filters.size();f++){

                sf << layer->filters[f].rows << " " << layer->filters[f].cols << "\n";

                for(int r = 0;r< layer->filters[f].rows;r++){
                    for(int c = 0;c<layer->filters[f].cols;c++){

                        sf << layer->filters[f].at<float>(r,c) << " ";
                    }
                    sf << "\n";
                }
            }

            //outInMapping
            sf << layer->OutInMapping.size() << std::endl;
            for(std::pair p:layer->OutInMapping){
                sf << p.first << " " << p.second << std::endl;
            }

        }
        else{
            std::cerr << "Fatal error while saving Network: LayerType <" << layerType << "> doesn't exist" << std::endl;
            assert(false);
        }
    }
}

ConvolutionalNetwork::ConvolutionalNetwork(std::string filename) {//TODO: make assert checks
    std::ifstream in(filename);
    in >> input_width;
    in >> input_heigth;
    in >> num_classes;
    in >> lambda;
    in >> propagations;
    int numLayers;
    in >> numLayers;

    layers = std::vector<NetworkLayer*>(numLayers);


    for(int l = 0;l<layers.size();l++){
        std::string layerType;
        in >> layerType;
        std::cout << std::endl;
        if(layerType == "MaxPool"){
            int s;
            int poolSize;
            in >> s;
            in >> poolSize;
            MaxPoolLayer *layer = new MaxPoolLayer(poolSize,s);//TODO:delete heap allocated variables on destruciton maybe dynamic pointer...
            //TODO: make destructor of ConvolutionalNetwork.
            layers[l] = layer;
        }
        else if(layerType == "ReLU"){
            ReLULayer *layer = new ReLULayer;
            layers[l] = layer;
        }
        else if(layerType == "FullyConnected"){

            std::string actFunc;
            in >> actFunc;

            int n;
            in >> n;



            //biases
            int biass;
            in >> biass;
            Mat1f bmat = Mat1f(biass,1);
            for(int y = 0;y<biass;y++){
                float biasval;
                in >> biasval;
                bmat.at<float>(y,0) = biasval;
            }


            //weights
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


            FullyConnectedLayer *layer = new FullyConnectedLayer(n,biass,lambda,actFunc);

            layer->biases = bmat.clone();
            layer->weights = currW.clone();
            layers[l] = layer;


        }
        else if(layerType == "Convolutional"){

            //filter size
            int fs;
            in >> fs;

            //num filters
            int nf;
            in >> nf;

            std::vector<Mat1f> filters;
            for(int f = 0;f<nf;f++){
                int rows;
                int cols;
                in >> rows >> cols;
                Mat1f filt(rows,cols);
                for(int r = 0; r< rows;r++){
                    for(int c = 0; c< cols;c++){
                        float val;
                        in >> val;
                        filt.at<float>(r,c)=val;

                    }
                }
                filters.push_back(filt.clone());
            }

            //OutInMapping
            int oim;
            in >> oim;

            for(int p = 0;p<oim;p++){
                std::pair<int, int> pr;
                //not sure whether it works
                in >> pr.first >> pr.second;
            }
            ConvolutionalLayer *layer = new ConvolutionalLayer(fs,nf,lambda);
            layer->filters = copyVec(filters);

            layers[l] = layer;

        }
        else{
            std::cerr << "Fatal error while loading Network on layer " << l <<" : LayerType <" << layerType << "> doesn't exist" << std::endl;


            assert(false);
        }
    }

}



