#include "ConvolutionalLayer.h"





std::vector<Mat1f> ConvolutionalLayer::use(std::vector<Mat1f> vec) {
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
