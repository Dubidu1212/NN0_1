#include "MaxPoolLayer.h"




std::vector<Mat1f> MaxPoolLayer::use(std::vector<Mat1f> in) {
    inputDimensionsRows = in[0].rows;
    inputDimensionsCols = in[0].cols;

    inputHistory = copyVec(in);
    std::vector<Mat1f> out;
    for(int m = 0;m<in.size();m++){
        Mat1f mat = in[m];
        float dimr = mat.rows;
        dimr =ceil( dimr/((float)s));
        float dimc = mat.cols;
        dimc =ceil( dimc/((float)s));

        Mat1f tempOut = Mat1f(dimr,dimc,0.0f);

        //iterations in said direction



        for(int r = 0,rit = 0;rit < (int)dimr;r+=s,rit++){//original >=
            for(int c = 0,cit = 0;cit < (int)dimc;c+=s,cit++){//original >=

                float max;
                max = mat.at<float>(r,c);
                for(int pr = 0;pr < poolSize;pr++){
                    for(int pc = 0;pc<poolSize;pc++){

                        float temp = mat.at<float>(std::min(r+pr,mat.rows-1),std::min(c+pc,mat.cols-1));

                        if(temp>max){
                            max = temp;
                        }

                    }
                }

                tempOut.at<float>(rit,cit) = max;
            }
        }
        out.push_back(tempOut.clone());
    }







    return out;
}

MaxPoolLayer::MaxPoolLayer(int poolingSize, int stride) {
    layerType = "MaxPool";
    s = stride;
    poolSize = poolingSize;
}

std::vector<Mat1f> MaxPoolLayer::dErr(std::vector<Mat1f> nextLayerError) {
    //might safe the positions of the maxima during use to then use them here later

    std::vector<Mat1f> out = copyDimVec(inputHistory);
    for(int m = 0;m<inputHistory.size();m++){
        Mat1f mat = inputHistory[m];
        float dimr = mat.rows;
        dimr =ceil( dimr/((float)s));
        float dimc = mat.cols;
        dimc =ceil( dimc/((float)s));


        for(int r = 0,rit = 0;rit < (int)dimr;r+=s,rit++){//original >=
            for(int c = 0,cit = 0;cit < (int)dimc;c+=s,cit++){//original >=

                float max;
                max = mat.at<float>(r,c);

                std::pair<int,int> maxCoordinates;
                maxCoordinates = {r,c};

                for(int pr = 0;pr < poolSize;pr++){
                    for(int pc = 0;pc<poolSize;pc++){

                        float temp = mat.at<float>(std::min(r+pr,mat.rows-1),std::min(c+pc,mat.cols-1));
                        if(temp > max){
                            max = temp;
                            maxCoordinates = {std::min(r+pr,mat.rows-1),std::min(c+pc,mat.cols-1)};

                        }
                    }
                }
                //set the value with the coordinates of the maximum to the error
                out[m].at<float>(maxCoordinates.first, maxCoordinates.second) = nextLayerError[m].at<float>(rit,cit);

            }
        }

    }




    return out;
}

void MaxPoolLayer::applyError() {
    //no error to apply
    return;
}

std::tuple<int, int, int> MaxPoolLayer::outputSize(std::tuple<int,int,int> in) {//not sure whether it works
    return std::make_tuple(std::ceil(std::get<0>(in)/2.0f),std::ceil(std::get<1>(in)/2.0f),std::get<2>(in));
}
