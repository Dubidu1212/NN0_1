#include <iostream>
#include <string>
#include "FeedForwardNetwork.h"
#include <fstream>
#include "FullyConnectedLayer.h"
#include "ConvolutionalNetwork.h"
#include "ConvolutionalLayer.h"
#include "ReLULayer.h"
#include "MaxPoolLayer.h"
#include <ctime>





int main() {

    cv::theRNG().state = std::time(nullptr);




    std::string usage;
    std::cout << "Choose an application for the network." << std::endl;
    std::cout << "Options are: " << "1:BinaryFeed, "<<"2:MNIST FeedForward, " << "3:load and use MNIST, " << "4:load and continue training MNIST, "<<"5:Use FeedForward in Convlolutional form on MNIST" << "\n6:Convolutional MNIST "<< " 7:Load and Train Convolutional MNIST" << std::endl;
    std::cout << "8: Load and use Convolutional Network" << std::endl;
    std::cin >> usage;
    if(usage == "1"){


        FeedForwardNetwork ffn({2,10,10,2},0.1);
        //ffn.print();

        for(int iteration = 0;iteration<50000;iteration++){
            int a,b;
            a = rand()%2;
            b = rand()%2;

            int out;
            if(a == b && a == 0){
                out = 0;
                ffn.trainSingle((Mat1f(2,1) << a,b),(Mat1f(2,1)<< 0,1));
            }
            else if(b == 0){
                ffn.trainSingle((Mat1f(2,1) << a,b),(Mat1f(2,1)<< 1,0));
            }
            else if(a == 0){
                ffn.trainSingle((Mat1f(2,1) << a,b),(Mat1f(2,1)<< 1,0));
            }
            else{
                ffn.trainSingle((Mat1f(2,1) << a,b),(Mat1f(2,1)<< 0,1));
            }


        }

        std::cout << "0,0 \n" << ffn.use(Mat1f(2,1,0.0)) << std::endl;
        std::cout << "1,1 \n" << ffn.use(Mat1f(2,1,1.0)) << std::endl;
        std::cout << "1,0 \n" << ffn.use((Mat1f(2,1) << 1,0)) << std::endl;
        std::cout << "0,1 \n" << ffn.use((Mat1f(2,1) << 0,1)) << std::endl;


    }
    else if(usage == "2"){
        std::vector<cv::Mat> vec;

        read_Mnist("../Resources/train-images.idx3-ubyte", vec);

        std::cout<<vec.size()<<std::endl;


        std::vector<double> labels(vec.size());
        read_Mnist_Label("../Resources/train-labels.idx1-ubyte", labels);
        FeedForwardNetwork ffn({784,128,64,10},0.07);


        std::cout << "percentage file name:\n" ;
        std::string pf;
        std::cin >> pf;
        std::ofstream percentagefile;
        percentagefile.open(pf);

        std::cout << "Number of epochs:\n";
        float epochs;
        std::cin >> epochs;

        std::cout << "saving to file:" << std::endl;
        std::string savefile;
        std::cin >> savefile;

        for(long long i = 0;i<vec.size()*epochs;i++){
            int random = rand()%(vec.size());
            Mat1f image = vec[random];
            Mat1f output = Mat1f(10,1,0.0f);
            output.at<float>(labels[random]) = 1;


            ffn.trainSingle(image.reshape(1,1).t(),output);
            if(i%1000==0){
                std::cout << i<< std::endl;

                //for show purposes
                if(i>=epochs*9){
                    for(int x = 0;x<5;x++){
                        int random = rand()%(vec.size());
                        Mat1f res= ffn.use(vec[random].reshape(1,1).t());
                        double min;
                        double max;
                        cv::Point min_loc, max_loc;
                        cv::minMaxLoc(res, &min, &max, &min_loc, &max_loc);
                        namedWindow("Predicted a:"+std::to_string(max_loc.y),WINDOW_NORMAL);
                        imshow("Predicted a:"+std::to_string(max_loc.y),vec[random]);
                        waitKey();
                    }
                }


                //testing the performance
                int right = 0;
                int wrong = 0;
                for(int x = 0;x< 5000;x++){
                    int random = rand()%(vec.size());
                    Mat1f res= ffn.use(vec[random].reshape(1,1).t());

                    double min;
                    double max;
                    cv::Point min_loc, max_loc;
                    cv::minMaxLoc(res, &min, &max, &min_loc, &max_loc);
                    if(max_loc.y == labels[random]){
                        right++;
                    }
                    else{
                        wrong++;
                    }
                }
                std::cout << "right: " << right << " " <<"wrong: " << wrong << std::endl;
                std::cout << ((float)right/((float)wrong+(float)right))*100 << "%" << std::endl;
                percentagefile <<  i << "," <<((float)right/((float)wrong+(float)right))*100  << "\n";


            }
        }
        percentagefile.close();

        ffn.save(savefile);


    }
    else if(usage == "3"){
        std::cout << "filename:\n";
        std::string file;
        std::cin >> file;

        FeedForwardNetwork loaded(file);


        std::vector<cv::Mat> vec;

        read_Mnist("../Resources/train-images.idx3-ubyte", vec);

        std::cout<<vec.size()<<std::endl;

        std::vector<double> labels(vec.size());
        read_Mnist_Label("../Resources/train-labels.idx1-ubyte", labels);

        for(int x = 0;x< 5000;x++){

            int random = rand()%(vec.size());

            Mat1f res = loaded.use(vec[random].reshape(1,1).t());
            double min;
            double max;
            cv::Point min_loc, max_loc;
            cv::minMaxLoc(res, &min, &max, &min_loc, &max_loc);
            std::cout << max_loc.y << std::endl;
            imshow(std::to_string(max_loc.y),vec[random]);

            waitKey();
        }
    }
    else if(usage =="4"){
        std::cout << "filename:\n";
        std::string file;
        std::cin >> file;

        FeedForwardNetwork ffn(file);



        std::vector<cv::Mat> vec;

        read_Mnist("../Resources/train-images.idx3-ubyte", vec);

        std::cout<<vec.size()<<std::endl;


        std::vector<double> labels(vec.size());
        read_Mnist_Label("../Resources/train-labels.idx1-ubyte", labels);


        std::cout << "percentage file name:\n" ;
        std::string pf;
        std::cin >> pf;
        std::ofstream percentagefile;
        percentagefile.open(pf,std::ios_base::app);

        std::cout << "Number of epochs:\n";
        float epochs;
        std::cin >> epochs;

        std::cout << "saving to file:" << std::endl;
        std::string savefile;
        std::cin >> savefile;

        for(long long i = 0;i<vec.size()*epochs;i++){
            int random = rand()%(vec.size());
            Mat1f image = vec[random];
            Mat1f output = Mat1f(10,1,0.0f);
            output.at<float>(labels[random]) = 1;


            ffn.trainSingle(image.reshape(1,1).t(),output);
            if(i%1000==0){
                std::cout << i<< std::endl;

                //testing the performance
                int right = 0;
                int wrong = 0;
                for(int x = 0;x< 5000;x++){
                    int random = rand()%(vec.size());
                    Mat1f res= ffn.use(vec[random].reshape(1,1).t());

                    double min;
                    double max;
                    cv::Point min_loc, max_loc;
                    cv::minMaxLoc(res, &min, &max, &min_loc, &max_loc);
                    if(max_loc.y == labels[random]){
                        right++;
                    }
                    else{
                        wrong++;
                    }
                }
                std::cout << "right: " << right << " " <<"wrong: " << wrong << std::endl;
                std::cout << ((float)right/((float)wrong+(float)right))*100 << "%" << std::endl;
                percentagefile <<  i << "," <<((float)right/((float)wrong+(float)right))*100  << "\n";


            }
        }
        percentagefile.close();

        ffn.save(savefile);
    }
    else if(usage == "5"){
        ConvolutionalNetwork convNet(28,28,10,0.07f,"MSE");

        convNet.layers.resize(3);

        FullyConnectedLayer layer0(128,784,0.07f,"Sigmoid");
        convNet.layers[0]=&layer0;

        FullyConnectedLayer layer1(64,128,0.07f,"Sigmoid");
        convNet.layers[1]=&layer1;

        FullyConnectedLayer layer2(10,64,0.07f,"Sigmoid");
        convNet.layers[2]=&layer2;

        std::vector<cv::Mat> vec;

        read_Mnist("../Resources/train-images.idx3-ubyte", vec);

        std::cout<<vec.size()<<std::endl;


        std::vector<double> labels(vec.size());
        read_Mnist_Label("../Resources/train-labels.idx1-ubyte", labels);

        std::cout << "savefile name:\n";
        std::string sf;
        std::cin >> sf;

        std::cout << "percentage file name:\n" ;
        std::string pf;
        std::cin >> pf;
        std::ofstream percentagefile;
        percentagefile.open(pf);

        std::cout << "Number of epochs:\n";
        float epochs;
        std::cin >> epochs;



        for(long long i = 0;i<vec.size()*epochs;i++){
            int random = rand()%(vec.size());
            Mat1f image = vec[random];
            Mat1f output = Mat1f(10,1,0.0f);
            output.at<float>(labels[random]) = 1;



            convNet.wholePropagation(image,output);

            if(i%32==0){
                convNet.applyError();
            }

            if(i%1000==0){

                std::cout << i<< std::endl;

                //testing the performance
                int right = 0;
                int wrong = 0;
                for(int x = 0;x< 5000;x++){
                    int random = rand()%(vec.size());


                    Mat1f res = convNet.use(vec[random]);

                    double min;
                    double max;
                    cv::Point min_loc, max_loc;
                    cv::minMaxLoc(res, &min, &max, &min_loc, &max_loc);
                    if(max_loc.y == labels[random]){
                        right++;
                    }
                    else{
                        wrong++;
                    }
                }
                std::cout << "right: " << right << " " <<"wrong: " << wrong << std::endl;
                std::cout << ((float)right/((float)wrong+(float)right))*100 << "%" << std::endl;
                percentagefile <<  i << "," <<((float)right/((float)wrong+(float)right))*100  << "\n";


            }
        }
        percentagefile.close();
        convNet.save(sf);

    }
    else if(usage == "6"){
        ConvolutionalNetwork convNet(28,28,10,0.07f,"MSE");

        convNet.layers.resize(7);

        ConvolutionalLayer layer0(3,8,0.07f);
        convNet.layers[0]= &layer0;

        ReLULayer layer1;
        convNet.layers[1]= &layer1;

        ConvolutionalLayer layer2(3,8,0.07f);
        convNet.layers[2]= &layer2;

        ReLULayer layer3;
        convNet.layers[3]= &layer3;

        MaxPoolLayer layer4(2,2);
        convNet.layers[4] = &layer4;


        FullyConnectedLayer layer5(128,9216,0.07f,"Sigmoid");
        convNet.layers[5]=&layer5;

        FullyConnectedLayer layer6(10,128,0.07f,"Sigmoid");
        convNet.layers[6]=&layer6;


        std::vector<cv::Mat> vec;

        read_Mnist("../Resources/train-images.idx3-ubyte", vec);

        std::cout<<vec.size()<<std::endl;


        std::vector<double> labels(vec.size());
        read_Mnist_Label("../Resources/train-labels.idx1-ubyte", labels);



        std::cout << "percentage file name:\n" ;
        std::string pf;
        std::cin >> pf;
        std::ofstream percentagefile;
        percentagefile.open(pf);


        std::cout << "savefile name:\n" ;
        std::string sf;
        std::cin >> sf;


        std::cout << "Number of epochs:\n";
        float epochs;
        std::cin >> epochs;



        for(long long i = 0;i<vec.size()*epochs;i++){


            int random = rand()%(vec.size());
            Mat1f image = vec[random];
            Mat1f output = Mat1f(10,1,0.0f);
            output.at<float>(labels[random]) = 1;



            convNet.wholePropagation((image/256)-0.5,output);

            if(i%32==0){
                convNet.applyError();
            }

            if(i%1000==0){

                std::cout << i << std::endl;





                //testing the performance
                int right = 0;
                int wrong = 0;
                for(int x = 0;x< 1000;x++){
                    int random = rand()%(vec.size());

                    Mat1f image = vec[random];
                    Mat1f res = convNet.use((image/256)-0.5);


                    double min;
                    double max;
                    cv::Point min_loc, max_loc;
                    cv::minMaxLoc(res, &min, &max, &min_loc, &max_loc);
                    if(max_loc.y == labels[random]){
                        right++;
                    }
                    else{
                        wrong++;
                    }
                }
                std::cout << "right: " << right << " " <<"wrong: " << wrong << std::endl;
                std::cout << ((float)right/((float)wrong+(float)right))*100 << "%" << std::endl;
                percentagefile <<  i << "," <<((float)right/((float)wrong+(float)right))*100  << "\n";


            }
        }
        percentagefile.close();
        convNet.save(sf);
    }
    else if(usage == "7") {
        std::cout << "Filename:" << std::endl;
        std::string fn;
        std::cin >> fn;
        ConvolutionalNetwork convNet(fn);


        std::cout << "percentage file name:\n";
        std::string pf;
        std::cin >> pf;
        std::ofstream percentagefile;
        percentagefile.open(pf);

        std::cout << "savefile name:\n";
        std::string sf;
        std::cin >> sf;


        std::cout << "Number of epochs:\n";
        float epochs;
        std::cin >> epochs;


        std::vector<cv::Mat> vec;

        read_Mnist("../Resources/train-images.idx3-ubyte", vec);

        std::cout << vec.size() << std::endl;


        std::vector<double> labels(vec.size());
        read_Mnist_Label("../Resources/train-labels.idx1-ubyte", labels);

        for (long long i = 0; i < vec.size() * epochs; i++) {


            int random = rand() % (vec.size());
            Mat1f image = vec[random];
            Mat1f output = Mat1f(10, 1, 0.0f);
            output.at<float>(labels[random]) = 1;


            convNet.wholePropagation((image / 256) - 0.5, output);

            if (i % 32 == 0) {
                convNet.applyError();
            }

            if (i % 1000 == 0) {

                std::cout << i << std::endl;





                //testing the performance
                int right = 0;
                int wrong = 0;
                for (int x = 0; x < 1000; x++) {
                    int random = rand() % (vec.size());

                    Mat1f image = vec[random];
                    Mat1f res = convNet.use((image / 256) - 0.5);


                    double min;
                    double max;
                    cv::Point min_loc, max_loc;
                    cv::minMaxLoc(res, &min, &max, &min_loc, &max_loc);
                    if (max_loc.y == labels[random]) {
                        right++;
                    } else {
                        wrong++;
                    }
                }
                std::cout << "right: " << right << " " << "wrong: " << wrong << std::endl;
                std::cout << ((float) right / ((float) wrong + (float) right)) * 100 << "%" << std::endl;
                percentagefile << i << "," << ((float) right / ((float) wrong + (float) right)) * 100 << "\n";


            }
        }
        percentagefile.close();
        convNet.save(sf);
    }
    else if(usage == "8"){
        std::cout << "Filename:" << std::endl;
        std::string fn;
        std::cin >> fn;
        ConvolutionalNetwork convNet(fn);

        std::vector<cv::Mat> vec;

        read_Mnist("../Resources/train-images.idx3-ubyte", vec);

        std::cout << vec.size() << std::endl;


        std::vector<double> labels(vec.size());
        read_Mnist_Label("../Resources/train-labels.idx1-ubyte", labels);

        float wrong=0,right= 0;
        for(int x = 0;x<100000;x++){


            int random = rand() % (vec.size());

            Mat1f image = vec[random];
            Mat1f res = convNet.use((image / 256) - 0.5);


            double min;
            double max;
            cv::Point min_loc, max_loc;
            cv::minMaxLoc(res, &min, &max, &min_loc, &max_loc);

            /*
            if (max_loc.y == labels[random]) {
                right++;
            } else {
                wrong++;
            }

            std::cout << right/(wrong+right)*100 << std::endl;
            */
            imshow(std::to_string(max_loc.y),vec[random]);
            waitKey();

        }

    }
    else{
        std::cout << "Sorry but: " << usage << " doesn't exist" << std::endl;
    }





    return 0;
}