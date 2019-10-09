#include "ConvolutionalLayer.h"





std::vector<Mat1f> ConvolutionalLayer::use(std::vector<Mat1f> vec) {


    return ownUse(vec);
    //initializes input dimension rows and cols
    inputDimensionsRows = vec[0].rows;
    inputDimensionsCols = vec[0].cols;

    inputHistory = copyVec(vec);
    std::vector<Mat1f> out;
    OutInMapping.resize(vec.size()*filters.size());
    int counter = 0;
    for(int m = 0;m<vec.size();m++){
        Mat1f in = vec[m];
        for(int f = 0;f<filters.size();f++){
            Mat1f res;
            filter2D(in,res,-1,filters[f]);

            out.push_back(res.clone());
            OutInMapping[counter] = std::make_pair(m,f);
            counter++;
        }


    }
    return out;

}

std::vector<Mat1f> ConvolutionalLayer::ownUse(std::vector<Mat1f> vec) {
    //initializes input dimension rows and cols
    inputDimensionsRows = vec[0].rows;
    inputDimensionsCols = vec[0].cols;

    inputHistory = copyVec(vec);
    OutInMapping.resize(vec.size()*filters.size());
    int counter = 0;
    std::vector<Mat1f> out = std::vector<Mat1f>(filters.size()*vec.size());

    for(int m = 0;m<vec.size();m++){
        Mat1f in = vec[m];
        for(int f = 0;f<filters.size();f++){
            Mat1f filter = filters[f];
            Mat1f res = Mat1f(in.rows-filterSize+1,in.cols-filterSize+1,0.0f);

            for(int r = 0;r<in.rows-filterSize+1;r++){
                for(int c = 0;c< in.cols-filterSize+1;c++){

                    float sum = 0;

                    for(int fr = 0;fr<filter.rows;fr++){
                        for(int fc = 0;fc<filter.cols;fc++){

                            sum += filter.at<float>(fr,fc)*in.at<float>(r+fr,c+fc);
                        }
                    }
                    res.at<float>(r,c) = sum;

                }
            }

            //out.push_back(res.clone());

            out[m*filters.size()+f] = res.clone();

            OutInMapping[counter] = std::make_pair(m,f);
            counter++;

        }
    }


    return out;
}

ConvolutionalLayer::ConvolutionalLayer(int filterSize, int numFilters, float lamdba) {
    layerType = "Convolutional";
    this->lambda = lamdba;
    this->filterSize = filterSize;
    //initialize filters
    filters.resize(numFilters);
    for(int f = 0;f<numFilters;f++){
        Mat1f filter = Mat1f(filterSize,filterSize,0.0f);
        randu(filter,Scalar(-1),Scalar(1));

        filters[f] = filter.clone();

    }

    //initialize errorAccumulate
    errorAccumulate = std::vector<Mat1f>(numFilters,Mat1f(filterSize,filterSize,0.0f));

}



Mat1f ConvolutionalLayer::ErrdInSingle(int outMat) {
    //this is probably very slow
    Mat1f filter = filters[OutInMapping[outMat].second];
    Mat1f in = inputHistory[OutInMapping[outMat].first];
    Mat1f out = Mat1f(in.rows,in.cols,0.0f);

    for(int r = 0;r < in.rows-filterSize+1;r++){
        for(int c = 0;c<in.cols-filterSize+1;c++){

            for(int fr = 0;fr<filterSize;fr++){
                for(int fc = 0;fc <filterSize;fc++){
                    out.at<float>(r+fr,c+fc) += filter.at<float>(fr,fc) * nextLayerErr[outMat].at<float>(r,c);
                }
            }

        }
    }


    return out;

    //return out.mul(nextLayerErr[outMat]);
}



std::vector<Mat1f> ConvolutionalLayer::dErr(std::vector<Mat1f> in){
    passes++;
    nextLayerErr = copyVec(in);//TODO: dont use next layer err but pass it to the functions instead




    std::vector<Mat1f> ErrdIn = std::vector<Mat1f>(OutInMapping.size()/filters.size(),Mat1f(inputDimensionsRows,inputDimensionsCols,0.0f));
    for(int outmat = 0;outmat<OutInMapping.size();outmat++){

        ErrdIn[OutInMapping[outmat].first] += ErrdInSingle(outmat);

    }


    ErrdFilter();

    return ErrdIn;
}

void ConvolutionalLayer::ErrdFilter(){



    for(int in = 0;in<inputHistory.size();in++){

        for(int f = 0;f<filters.size();f++){
            Mat1f inMat = inputHistory[in];
            Mat1f error = nextLayerErr[in*filters.size()+f].clone();//currently affected error

            for(int r = 0;r<inMat.rows-filterSize+1;r++){
                for(int c = 0;c<inMat.cols-filterSize+1;c++){


                    for(int fr = 0;fr<filterSize;fr++){
                        for(int fc = 0;fc<filterSize;fc++){

                            errorAccumulate[f].at<float>(fr,fc) += inMat.at<float>(r+fr,c+fc)*error.at<float>(r,c);
                        }
                    }

                }
            }


        }
    }
}

void ConvolutionalLayer::applyError() {


    for(int f = 0;f<filters.size();f++){


        /*
        namedWindow("filters",WINDOW_NORMAL);
        imshow("filters",(filters[f]+0.5));
        resizeWindow("filters",400,400);
        waitKey(0)*/

        filters[f] -= (errorAccumulate[f]/(passes*filterSize*filterSize))*lambda;

        /*
        double min, max;
        cv::minMaxLoc(filters[f], &min, &max);

        //keeps values between -1 and 1
        filters[f]/=std::max(-min,max);*/

        /*
        namedWindow("filters",WINDOW_NORMAL);
        imshow("filters",(filters[f]+0.5));
        resizeWindow("filters",400,400);
        waitKey(0);*/



    }


    errorAccumulate = std::vector<Mat1f>(filters.size(),Mat1f(filterSize,filterSize,0.0f));
    passes=0;



}

std::tuple<int, int, int> ConvolutionalLayer::outputSize(std::tuple<int, int, int> in) {

    return std::make_tuple(std::get<0>(in)-(this->filterSize-1),std::get<1>(in)-(this->filterSize-1),std::get<2>(in)*this->filters.size());
}
