#include "gene.h"

gene::gene(int lenght, int numClasses,int exclusion) {
    this->exclusion = exclusion;
    this->lenght = lenght;
    this->numClasses = numClasses;
    bool D2 = true;
    for(int a = 0;a<lenght-1;a++){

        alleles.push_back(generateRandomAllele(D2));
        if(alleles[a].at(0)=='f'){//newest layer is fully connected
            D2 = false;
        }
    }
    alleles.emplace_back("fCL");//fully connected last
}

gene gene::randomize(float probabilityL, float probability) {

}

gene::gene(gene p1, gene p2) {//only works for p1.length=p2.length
    assert(p1.lenght == p2.lenght);

    this->lenght = p1.lenght;
    this->numClasses = p1.numClasses;
    for(int x = 0;x<p1.lenght;x++){
        bool P1 = rand()%2;//TODO:make more random
        if(P1){
            this->alleles.emplace_back(p1.alleles[x]);
        }
        else{
            this->alleles.emplace_back(p2.alleles[x]);
        }
    }

}
/*Allele language
 *
 *first two charaters = layer type
 * fc = fully connected
 * co = convolutional
 * mp = maxpool
 * rl = relu
 *
 *Number until character ;
 * if convolutional another number follows;(width and height)
 * if maxpool another number follows;(poolsize,stride)
 *
 *For fully connected activation function
 * si = sigmoid
 * sm = softmax
 */


std::string gene::generateRandomAllele(bool D2){
    std::string allele;
    if(D2){//all layers
        int type = rand()%(4-exclusion);
        if(type == 0){//fully connected
            allele += "fc";
            int numNeurons = rand()%randRangeFC;//Number of neurons which are maximally randRangeFC
            numNeurons+=1;//So we dont have a layer with 0 nodes;
            allele += std::to_string(numNeurons);
            allele += ";";
            if(rand()%2){
                allele+="si";
            }
            else{
                allele+="sm";//might want to reduce probabillity or only allow in last layer
            }

        }
        else if(type == 1){//maxpool
            allele += "mp";
        }
        else if(type == 2){//relu
            allele += "rl";
        }
        else if(type == 3){//conv
            allele += "co";
        }
    }
    else{//only fully connected layers

    }
    return std::__cxx11::string();
}
