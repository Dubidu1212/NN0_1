#include "helperFunctions.h"


void addNumToVec(std::vector<Mat1f> in, float num){
    for(Mat1f m : in){
        for(int row = 0; row < m.rows; ++row) {
            for(int col = 0; col < m.cols; ++col) {
                m.at<float>(row,col) += num;
            }
        }
    }
}


void eWMOp(Mat1f &in, float (*op)(float)) {
    for(int row = 0; row < in.rows; ++row) {
        for(int col = 0; col < in.cols; ++col) {
            in.at<float>(row,col) = op(in.at<float>(row,col));
        }
    }
}


float sigmoid(float in) {
    return 1/(1+exp(in*-1));
}

float sigmoidPrime(float in) {
    return sigmoid(in)*(1-sigmoid(in));
}

std::vector<Mat1f> copyVec(std::vector<Mat1f> in) {
    //TODO: make pointer
    std::vector<Mat1f> retVec = std::vector<Mat1f>(in.size());
    for(int x = 0;x<in.size();x++){

        retVec[x] = in[x].clone();
    }

    return retVec;
}

std::vector<Mat1f> copyDimVec(std::vector<Mat1f> in,float fillVal) {
    std::vector<Mat1f> retVec;
    for(int i = 0;i<in.size();i++){
        retVec.emplace_back(Mat1f(in[i].rows,in[i].cols,fillVal));//maybe push_back
    }
    return retVec;
}

float ReLU(float in) {
    return std::max(in,0.0f);
}

//TODO: this is worng because relu is not dependant on output but on input!
float dReLU(float in) {
    if(in < 0){
        return 0;
    }
    else if(in == 0){
        return 0.5;//this value is arbitrary
    }
    else{
        return 1;
    }
}

int ReverseInt (int i){
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_Mnist(std::string filename, std::vector<cv::Mat> &vec){

    std::ifstream file (filename, std::ios::binary);

    if (file.is_open()){
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;

        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);

        for(int i = 0; i < number_of_images; ++i){
            cv::Mat tp = Mat::zeros(n_rows, n_cols, CV_8UC1);
            for(int r = 0; r < n_rows; ++r){
                for(int c = 0; c < n_cols; ++c){
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    tp.at<uchar>(r, c) = (int) temp;
                }
            }
            vec.push_back(tp);
        }
    }
}

void read_Mnist_Label(std::string filename, std::vector<double> &vec)
{
    std::ifstream file (filename, std::ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        for(int i = 0; i < number_of_images; ++i)
        {
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            vec[i]= (double)temp;
        }
    }
}


