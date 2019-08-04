#include "MaxPoolLayer.h"

std::vector<Mat1f> MaxPoolLayer::use(std::vector<Mat1f> in) {
    std::vector<Mat1f> out;
    for(int m = 0;m<in.size();m++){
        Mat1f mat = in[m];
        float dimr = mat.rows;
        dimr =ceil( dimr/(float)s);
        float dimc = mat.cols;
        dimc =ceil( dimc/(float)s);

        Mat1f tempOut = Mat1f(dimr,dimc);
        for(int r = 0;(r*s)+poolSize >= mat.rows;r++){

            for(int c = 0;(c*s)+poolSize >= mat.cols;c++){
                float max = std::numeric_limits<float>::min();
                for(int pr = 0;pr < poolSize;pr++){
                    for(int pc = 0;pc<poolSize;pc++){
                        max = std::max(max,mat.at<float>(r+pr,c+pc));
                    }
                }

                tempOut.at<float>(r,c) = max;
            }
        }
        out.push_back(mat.clone());
    }

    return out;
}

MaxPoolLayer::MaxPoolLayer(int poolingSize, int stride) {
    s = stride;
    poolSize = poolingSize;
}
