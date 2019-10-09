//
// Created by raphael on 27.08.19.
//

#include "DropoutLayer.h"

std::vector<Mat1f> DropoutLayer::use(std::vector<Mat1f> in) {
    if(training){



        for(int m = 0;m<in.size();m++){

            //reload DropoutMats
            if(passes%reloadPeriod == 0){//maybe setup on first use

                int a,b,c;
                std::tie(a,b,c) = inputDim;

                Mat1f dO = Mat1f(a,b,1);

                //fast method: Causes deviations in how many are set to 0 due to positions being set to 0 twice
                //dropoff will be logarithmic
                for(int x = 0;x<(a*b*(dropoutPercentage/100));x++){

                    int r,c;
                    r = distr(mt);
                    c = distc(mt);

                    dO.at<float>(r,c) = 0.0f;
                }
                DropoutMats[m] = dO.clone();

                /*
                for(int r = 0;r<in[m].rows;r++){
                    for(int c = 0;c<in[m].cols;c++){


                        //might want to save as mat1f and mul;
                    }
                }*/
            }

            //apply dropout
            in[m] = in[m].mul(DropoutMats[m]);

        }
        passes++;
    }
    else{
        for(int m = 0;m<in.size();m++){//for each mat in in
            //TODO: check for correctness
            in[m]*=((100-dropoutPercentage)/100);//make so layer doesnt output too much when during use

        }
    }


    return in;//maybe clone
}

std::vector<Mat1f> DropoutLayer::dErr(std::vector<Mat1f> in) {
    for(int m = 0;m<in.size();m++){
        in[m]= in[m].mul(DropoutMats[m]);

    }

    return in;//maybe clone
}

std::tuple<int, int, int> DropoutLayer::outputSize(std::tuple<int, int, int> in) {
    return in;
}

void DropoutLayer::applyError() {//no error to apply here
    return;
}

DropoutLayer::DropoutLayer(int relTime,float dropoutP,std::tuple<int,int,int> inputSize) {

    inputDim = inputSize;

    int a,b,c;
    std::tie(a,b,c) = inputSize;
    std::cout << a << " " << b << " " << c << std::endl;





    reloadPeriod = relTime;
    dropoutPercentage = dropoutP;
    DropoutMats = std::vector<Mat1f>(c);
    layerType = "Dropout";

    std::random_device rd;
    mt = std::mt19937(rd());
    //assumes inclusive , inclusive
    distr = std::uniform_int_distribution(0,a-1);
    distc = std::uniform_int_distribution(0,b-1);
}

