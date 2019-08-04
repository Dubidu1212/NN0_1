#include "ConvolutionalLayer.h"





std::vector<Mat1f> ConvolutionalLayer::use(std::vector<Mat1f> vec) {
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
    std::vector<Mat1f> out;
    for(int m = 0;m<vec.size();m++){
        Mat1f in = vec[m];
        for(int f = 0;f<filters.size();f++){
            Mat1f filter = filters[f];
            Mat1f res = Mat1f(in.rows-filterSize+1,in.cols-filterSize+1);

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
            out.push_back(res.clone());
            OutInMapping[counter] = std::make_pair(m,f);
            counter++;

        }
    }

    return out;
}

ConvolutionalLayer::ConvolutionalLayer(int filterSize, int numFilters) {
    //initialize filters
    for(int f = 0;f<numFilters;f++){
        Mat1f filter = Mat1f(filterSize,filterSize);
        randu(filter,Scalar(-1),Scalar(1));
        filters.push_back(filter.clone());
    }

    //initialize errorAccumulate
    errorAccumulate = std::vector<Mat1f>(numFilters);

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
                    out.at<float>(r+fr,c+fc) += filter.at<float>(fr,fc);
                }
            }

        }
    }

    //includes err
    return out.mul(nextLayerErr[outMat]);
}



std::vector<Mat1f> ConvolutionalLayer::dErr(std::vector<Mat1f> in){

    std::vector<Mat1f> ErrdIn = std::vector<Mat1f>(OutInMapping.size()/filters.size(),Mat1f(inputDimensionsRows,inputDimensionsCols,0.0f));
    for(int outmat = 0;outmat<OutInMapping.size();outmat++){

        ErrdIn[OutInMapping[outmat].first] += ErrdInSingle(outmat);

    }


    ErrdFilter();

    return ErrdIn;
}

void ConvolutionalLayer::ErrdFilter(){
    std::vector<Mat1f> out;
    for(int f = 0;f<filters.size();f++){
        Mat1f err = nextLayerErr[f];
        for(int mat = 1;mat<inputHistory.size();mat++){
            //f+mat*filters.size() = all outputmats processed with this filter

            err += nextLayerErr[f+mat*filters.size()];
        }
        //filter error
        Mat1f ferr = Mat1f(filterSize,filterSize);
        //can convolute the filter over the error saving to the filter all the errors it touched summing
        //probaly slow
        //can combine with OutdIn to make it more efficent (*2)
        for(int r = 0;r<err.rows-filterSize+1;r++){
            for(int c = 0;c<err.cols-filterSize+1;c++){

                for(int fr = 0;fr<filterSize;fr++){
                    for(int fc = 0;fc<filterSize;fc++){
                        ferr.at<float>(fr,fc) += err.at<float>(r+fr,c+fc);
                    }
                }

            }
        }
        errorAccumulate[f] += ferr;//maybe clone

    }

}

void ConvolutionalLayer::applyError() {
    for(int f = 0;f<filters.size();f++){
        filters[f]+= errorAccumulate[f]*lambda;
    }
    errorAccumulate = std::vector<Mat1f>(filters.size(),Mat1f(filterSize,filterSize,0.0f));

}
