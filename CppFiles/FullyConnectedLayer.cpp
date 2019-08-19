#include "FullyConnectedLayer.h"

std::vector<Mat1f> FullyConnectedLayer::use(std::vector<Mat1f> in) {
    //make a mathematical Vector(flatten and combine)
    inputDimensions = copyDimVec(in);

    Mat1f vec = Mat1f(1,0);
    for(Mat1f mat: in){

        hconcat(vec,mat.reshape(1,1),vec);
    }
    vec = vec.t();

    //set nodeHistory to the n or the (flatened) output of the last layer
    nodeHistory = vec.clone();

    std::vector<Mat1f> out;

    //casts the mat expression to a mat and clones it
    //the formula is: w*(n+b) where n = vec
    Mat1f outMat = weights*(vec+biases);


    lastpreSig = outMat.clone();

    //applying the activation function
    if(activationFunction == "Sigmoid"){
        //taking the sigmoid of each element
        eWMOp(outMat,sigmoid);
    }
    else if(activationFunction == "SoftMaxE" || activationFunction == "SoftMaxS"){//TODO: make numerically stable
        exp(outMat,outMat);//raises e to outMat
        float softsum = sum(outMat)[0];//only works with 1 chanel
        outMat=outMat/softsum;
        lastSoftmax = outMat.clone();
    }
    else{
        std::cerr << "Fatal error while applying activation: activation function <" << activationFunction << ">doesn't exist!" << std::endl;
    }




    out.push_back(outMat.clone());


    return out;
}



FullyConnectedLayer::FullyConnectedLayer(int nodes,int inputSize,float lambda,std::string activationFunction){
    this->activationFunction = activationFunction;
    layerType = "FullyConnected";
    this->lambda = lambda;
    this->nodes = nodes;
    weights = Mat1f(nodes,inputSize);
    randu(weights,Scalar(-1),Scalar(1));

    weightsError = Mat1f(nodes,inputSize,0.0f);

    biases = Mat1f(inputSize,1);
    randu(biases,Scalar(-1),Scalar(1));
    biasesError = Mat1f(inputSize,1,0.0f);
}

std::vector<Mat1f> FullyConnectedLayer::dErr(std::vector<Mat1f> in) {
    passes++;

    Mat1f vec = in[0];//possible beacuse error can only come from another fullyconnected layer



    //differentiate the activation function

    if(activationFunction == "Sigmoid"){
        //lastpreSig now becomes d(postsig)/d(presig)
        eWMOp(lastpreSig,sigmoidPrime);
    }
    else if(activationFunction == "SoftMaxS"){//Slow softmax without included error
        Mat1f tempdSoftmax(lastSoftmax.total(),1,0.0f);
        for(int i = 0;i< lastSoftmax.total();i++){
            for(int j = 0;j< lastSoftmax.total();j++){
                //maybe -=
                if(i != j){
                    tempdSoftmax.at<float>(i) += -lastSoftmax.at<float>(j)*lastSoftmax.at<float>(j);
                }
                else{
                    tempdSoftmax.at<float>(i) += (1-lastSoftmax.at<float>(j))*lastSoftmax.at<float>(j);
                }
            }
        }
        lastpreSig = tempdSoftmax.clone();
    }
    else if(activationFunction == "SoftMaxE"){//Softmax with included error

        lastpreSig = ((Mat1f)(lastSoftmax-vec)).clone();
    }
    else{
        std::cerr << "Fatal error while differentiating: activation function <" << activationFunction << ">doesn't exist!" << std::endl;
        assert(false);
    }







    //multiplies by d(err)/d(postsig) resulting in d(err)/d(presig)
    lastpreSig = lastpreSig.mul(vec);



    Mat1f outMat = weights.t() * lastpreSig;
    biasesError += weights.t() * lastpreSig;

    weightsError += lastpreSig* (biases+nodeHistory).t();




    std::vector<Mat1f> out;
    outMat = outMat.reshape(1,inputDimensions[0].rows*inputDimensions.size());//all inputmatrices stacked on top of eachother
    for(int m = 0;m<inputDimensions.size();m++){
        //only works if all images are the same size (probably is true)
        Rect matCutter(0,m*inputDimensions[m].rows,inputDimensions[m].cols,inputDimensions[m].rows);
        Mat1f tempMat = Mat1f(outMat,matCutter);
        out.push_back(tempMat.clone());
    }



    return out;
}

void FullyConnectedLayer::applyError() {
    //maybe +=

    weights -= (weightsError/passes)*lambda;
    weightsError = Mat1f(weights.rows,weights.cols,0.0f);


    biases -= (biasesError/passes)*lambda;
    biasesError = Mat1f(biases.rows,biases.cols,0.0f);
    passes = 0;



}