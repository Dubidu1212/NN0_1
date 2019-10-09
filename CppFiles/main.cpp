#include <iostream>
#include <string>
#include "FeedForwardNetwork.h"
#include <fstream>
#include "FullyConnectedLayer.h"
#include "ConvolutionalNetwork.h"
#include "ConvolutionalLayer.h"
#include "ReLULayer.h"
#include "MaxPoolLayer.h"
#include "DropoutLayer.h"
#include <ctime>






int main(int argc,char* argv[]) {




    cv::theRNG().state = std::time(nullptr);//maybe remove...

    //random stuff
    std::random_device rd;
    std::mt19937 mt(rd());


    std::string usage;

    if(argc == 1){//no commandline arguments
        std::cout << "Choose an application for the network." << std::endl;
        std::cout << "Options are: " << "1:BinaryFeed, "<<"2:MNIST FeedForward, " << "3:load and use MNIST, " << "4:load and continue training MNIST, "<<"5:Use FeedForward in Convlolutional form on MNIST" << "\n6:Convolutional MNIST "<< " 7:Load and Train Convolutional MNIST" << std::endl;
        std::cout << "8: Load and use Convolutional Network" << " 9:Use Convolutional Network on LISC " <<"10:Network building wizard" <<std::endl;
        std::cout << "11: Conv on WBC" << " 12: Load WBC and train"<< " 13: Use WBC"<<" 14: Use BloodCell" <<std::endl;
        std::cin >> usage;
    }
    else{//only takes first commandline argument
        usage = argv[0];
    }


    if(usage == "1"){


        FeedForwardNetwork ffn({2,10,10,2},0.1);
        //ffn.print();

        std::uniform_int_distribution<int> dist2(0,1);

        for(int iteration = 0;iteration<50000;iteration++){
            int a,b;

            a = dist2(mt);
            b = dist2(mt);

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


        std::uniform_int_distribution distV(0,(int)vec.size()-1);

        for(long long i = 0;i<vec.size()*epochs;i++){
            int random = distV(mt);
            Mat1f image = vec[random];
            Mat1f output = Mat1f(10,1,0.0f);
            output.at<float>(labels[random]) = 1;


            ffn.trainSingle(image.reshape(1,1).t(),output);
            if(i%1000==0){
                std::cout << i<< std::endl;

                //for show purposes
                if(i>=epochs*9){
                    for(int x = 0;x<5;x++){
                        int random = distV(mt);
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
                    int random = distV(mt);
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


        std::uniform_int_distribution<int> distV(0,(int)vec.size()-1);

        for(int x = 0;x< 5000;x++){

            int random = distV(mt);

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


        std::uniform_int_distribution<int> distV(0,(int)vec.size()-1);

        for(long long i = 0;i<vec.size()*epochs;i++){
            int random = distV(mt);
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
                    int random = distV(mt);
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
        convNet.layers[0] = std::make_unique<FullyConnectedLayer>(layer0);


        FullyConnectedLayer layer1(64,128,0.07f,"Sigmoid");
        convNet.layers[1] =std::make_unique<FullyConnectedLayer>(layer1);

        FullyConnectedLayer layer2(10,64,0.07f,"Sigmoid");
        convNet.layers[2] =std::make_unique<FullyConnectedLayer>(layer2);

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

        std::uniform_int_distribution<int> distV(0,(int)vec.size()-1);


        for(long long i = 0;i<vec.size()*epochs;i++){
            int random = distV(mt);
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
                    int random = distV(mt);


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
        convNet.layers[0]= std::make_unique<ConvolutionalLayer>(layer0);

        ReLULayer layer1;
        convNet.layers[1]= std::make_unique<ReLULayer>(layer1);

        ConvolutionalLayer layer2(3,8,0.07f);
        convNet.layers[2]= std::make_unique<ConvolutionalLayer>(layer2);

        ReLULayer layer3;
        convNet.layers[3]= std::make_unique<ReLULayer>(layer3);

        MaxPoolLayer layer4(2,2);
        convNet.layers[4] = std::make_unique<MaxPoolLayer>(layer4);


        FullyConnectedLayer layer5(128,9216,0.07f,"Sigmoid");
        convNet.layers[5]=std::make_unique<FullyConnectedLayer>(layer5);

        FullyConnectedLayer layer6(10,128,0.07f,"Sigmoid");
        convNet.layers[6]=std::make_unique<FullyConnectedLayer>(layer6);


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

        std::uniform_int_distribution<int> distV(0,(int)vec.size()-1);

        for(long long i = 0;i<vec.size()*epochs;i++){


            int random = distV(mt);
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
                    int random = distV(mt);

                    Mat1f subim = vec[random];
                    Mat1f res = convNet.use((subim/256)-0.5);


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


        std::uniform_int_distribution<int> distV(0,(int)vec.size()-1);

        for (long long i = 0; i < vec.size() * epochs; i++) {


            int random = distV(mt);
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
                    int random = distV(mt);

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

        //read_Mnist("../Resources/train-images.idx3-ubyte", vec);
        read_Mnist("../Resources/t10k-images.idx3-ubyte", vec);

        std::cout << vec.size() << std::endl;


        std::vector<double> labels(vec.size());
        //read_Mnist_Label("../Resources/train-labels.idx1-ubyte", labels);
        read_Mnist_Label("../Resources/t10k-labels.idx1-ubyte", labels);


        std::uniform_int_distribution<int> distV(0,(int)vec.size()-1);

        float wrong=0,right= 0;
        for(int x = 0;x<100000;x++){


            int random = distV(mt);

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

            std::cout << right/(wrong+right)*100 << std::endl;



        }

    }
    else if(usage == "9"){


        std::cout << "For Ground truth, enter G \n for Images enter I" << std::endl;
        char c;
        std::cin >> c;
        if(c =='G'){
            //temporary
            std::vector<Mat1f> TBaso;
            int BasoNum = 53;
            TBaso.resize(BasoNum);

            std::vector<Mat1f> TEosi;
            int EosiNum = 39;
            TEosi.resize(EosiNum);


            std::vector<Mat1f> TLymp;
            int LympNum = 52;
            TLymp.resize(LympNum);

            std::vector<Mat1f> TMono;
            int MonoNum = 48;
            TMono.resize(MonoNum);

            std::vector<Mat1f> TNeut;
            int NeutNum = 50;
            TNeut.resize(NeutNum);


            //real
            std::vector<Mat1f> Baso;
            std::vector<Mat1f> Eosi;
            std::vector<Mat1f> Lymp;
            //std::vector<Mat1f> Mixt;
            std::vector<Mat1f> Mono;
            std::vector<Mat1f> Neut;



            for(int b = 1;b<BasoNum+1;b++){
                Mat temp= ((imread("../Resources/LISC Database/Ground Truth Segmentation/Baso/areaforexpert1/"+std::to_string(b)+"_expert.bmp",IMREAD_GRAYSCALE)));

                resize(temp,temp,Size(178,142));
                //720*576
                temp.convertTo(TBaso[b-1],CV_32FC2);
                TBaso[b-1]= TBaso[b-1].clone()/256;



            }
            for(int e = 1;e<EosiNum+1;e++){
                Mat temp= ((imread("../Resources/LISC Database/Ground Truth Segmentation/eosi/areaforexpert1/"+std::to_string(e)+"_expert.bmp",IMREAD_GRAYSCALE)));

                resize(temp,temp,Size(178,142));
                //720*576
                temp.convertTo(TEosi[e-1],CV_32FC2);
                TEosi[e-1] = TEosi[e-1].clone()/256;

                //imshow("test",Eosi[e-1]);
                //waitKey();
            }

            for(int b = 1;b<LympNum+1;b++){
                Mat temp= ((imread("../Resources/LISC Database/Ground Truth Segmentation/lymp/areaforexpert1/"+std::to_string(b)+"_expert.bmp",IMREAD_GRAYSCALE)));

                resize(temp,temp,Size(178,142));
                //720*576
                temp.convertTo(TLymp[b-1],CV_32FC2);
                TLymp[b-1]= TLymp[b-1].clone()/256;



            }
            for(int e = 1;e<MonoNum+1;e++){
                Mat temp= ((imread("../Resources/LISC Database/Ground Truth Segmentation/mono/areaforexpert1/"+std::to_string(e)+"_expert.bmp",IMREAD_GRAYSCALE)));

                resize(temp,temp,Size(178,142));
                //720*576
                temp.convertTo(TMono[e-1],CV_32FC2);
                TMono[e-1] = TMono[e-1].clone()/256;

                //imshow("test",Mono[e-1]);
                //waitKey();
            }

            for(int b = 1;b<NeutNum+1;b++){
                Mat temp= ((imread("../Resources/LISC Database/Ground Truth Segmentation/neut/areaforexpert1/"+std::to_string(b)+"_expert.bmp",IMREAD_GRAYSCALE)));

                resize(temp,temp,Size(178,142));
                //720*576
                temp.convertTo(TNeut[b-1],CV_32FC2);
                TNeut[b-1]= TNeut[b-1].clone()/256;

            }



            //flip images
            BasoNum*=4;
            for(int b = 0;b<TBaso.size();b++){
                Baso.push_back(TBaso[b].clone());

                Mat1f temp = TBaso[b].clone();//clone to be safe
                flip(temp,temp,0);
                Baso.push_back(temp.clone());

                temp = TBaso[b].clone();
                flip(temp,temp,+1);
                Baso.push_back(temp.clone());

                temp = TBaso[b].clone();
                flip(temp,temp,-1);
                Baso.push_back(temp.clone());

            }

            EosiNum*=4;
            for(int b = 0;b<TEosi.size();b++){
                Eosi.push_back(TEosi[b].clone());

                Mat1f temp = TEosi[b].clone();//clone to be safe
                flip(temp,temp,0);
                Eosi.push_back(temp.clone());

                temp = TEosi[b].clone();
                flip(temp,temp,+1);
                Eosi.push_back(temp.clone());

                temp = TEosi[b].clone();
                flip(temp,temp,-1);
                Eosi.push_back(temp.clone());

            }


            LympNum*=4;
            for(int b = 0;b<TLymp.size();b++){
                Lymp.push_back(TLymp[b].clone());

                Mat1f temp = TLymp[b].clone();//clone to be safe
                flip(temp,temp,0);
                Lymp.push_back(temp.clone());

                temp = TLymp[b].clone();
                flip(temp,temp,+1);
                Lymp.push_back(temp.clone());

                temp = TLymp[b].clone();
                flip(temp,temp,-1);
                Lymp.push_back(temp.clone());

            }

            MonoNum*=4;
            for(int b = 0;b<TMono.size();b++){
                Mono.push_back(TMono[b].clone());

                Mat1f temp = TMono[b].clone();//clone to be safe
                flip(temp,temp,0);
                Mono.push_back(temp.clone());

                temp = TMono[b].clone();
                flip(temp,temp,+1);
                Mono.push_back(temp.clone());

                temp = TMono[b].clone();
                flip(temp,temp,-1);
                Mono.push_back(temp.clone());

            }

            NeutNum*=4;
            for(int b = 0;b<TNeut.size();b++){
                Neut.push_back(TNeut[b].clone());

                Mat1f temp = TNeut[b].clone();//clone to be safe
                flip(temp,temp,0);
                Neut.push_back(temp.clone());

                temp = TNeut[b].clone();
                flip(temp,temp,+1);
                Neut.push_back(temp.clone());

                temp = TNeut[b].clone();
                flip(temp,temp,-1);
                Neut.push_back(temp.clone());

            }




            //shuffle
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

            shuffle (Eosi.begin(), Eosi.end(), std::default_random_engine(seed));
            shuffle (Baso.begin(), Baso.end(), std::default_random_engine(seed));
            shuffle (Lymp.begin(), Lymp.end(), std::default_random_engine(seed));
            shuffle (Mono.begin(), Mono.end(), std::default_random_engine(seed));
            shuffle (Neut.begin(), Neut.end(), std::default_random_engine(seed));




            ConvolutionalNetwork convNet(178,142,2,0.07f,"MSE");

            ConvolutionalLayer c0(3,4,0.07f);
            convNet.layers.push_back(std::make_unique<ConvolutionalLayer>(c0));



            ReLULayer rl0;
            convNet.layers.push_back(std::make_unique<ReLULayer>(rl0));

            MaxPoolLayer m0(2,2);
            convNet.layers.push_back(std::make_unique<MaxPoolLayer>(m0));

            ConvolutionalLayer c1(3,4,0.07f);
            convNet.layers.push_back(std::make_unique<ConvolutionalLayer>(c1));

            ReLULayer rl1;
            convNet.layers.push_back(std::make_unique<ReLULayer>(rl1));

            MaxPoolLayer m1(2,2);
            convNet.layers.push_back(std::make_unique<MaxPoolLayer>(m1));

            ConvolutionalLayer c2(3,2,0.07f);
            convNet.layers.push_back(std::make_unique<ConvolutionalLayer>(c2));

            ReLULayer rl2;
            convNet.layers.push_back(std::make_unique<ReLULayer>(rl2));



            int a,b,c;
            std::tie(a,b,c) = convNet.outputDim();
            //44352
            FullyConnectedLayer f(8192,a*b*c,0.07f,"Sigmoid");
            convNet.layers.push_back(std::make_unique<FullyConnectedLayer>(f));

            FullyConnectedLayer f0(2048,8192,0.07f,"Sigmoid");
            convNet.layers.push_back(std::make_unique<FullyConnectedLayer>(f0));

            FullyConnectedLayer f1(512,2048,0.07f,"Sigmoid");
            convNet.layers.push_back(std::make_unique<FullyConnectedLayer>(f1));

            FullyConnectedLayer f2(128,512,0.07f,"Sigmoid");
            convNet.layers.push_back(std::make_unique<FullyConnectedLayer>(f2));


            DropoutLayer dO(4,25,convNet.outputDim());
            convNet.layers.push_back(std::make_unique<DropoutLayer>(dO));


            FullyConnectedLayer f3(32,128,0.07f,"Sigmoid");
            convNet.layers.push_back(std::make_unique<FullyConnectedLayer>(f3));

            FullyConnectedLayer f4(8,32,0.07f,"Sigmoid");
            convNet.layers.push_back(std::make_unique<FullyConnectedLayer>(f4));

            FullyConnectedLayer f5(5,8,0.07f,"Sigmoid");
            convNet.layers.push_back(std::make_unique<FullyConnectedLayer>(f5));


            int samples;
            std::cout << "Number of samples\n";
            std::cin >> samples;

            std::cout << "percentage file name:\n";
            std::string pf;
            std::cin >> pf;
            std::ofstream percentagefile;
            percentagefile.open(pf);

            std::cout << "savefile name:\n";
            std::string sf;
            std::cin >> sf;

            std::uniform_int_distribution<int> dist2(0,4);

            //-40 so i can use 39 for testing overfitting
            std::uniform_int_distribution<int> distBaso(40,BasoNum-1);
            std::uniform_int_distribution<int> distEosi(40,EosiNum-1);
            std::uniform_int_distribution<int> distLymp(40,LympNum-1);
            std::uniform_int_distribution<int> distMono(40,MonoNum-1);
            std::uniform_int_distribution<int> distNeut(40,NeutNum-1);





            std::uniform_int_distribution<int> distTest(0,39);

            for(int n = 0;n<samples;n++){


                if(dist2(mt) == 0){//baso
                    int random = distBaso(mt);
                    Mat1f in = Baso[random];
                    Mat1f out = Mat1f(5,1,0.0f);
                    out.at<float>(0) = 1;
                    convNet.wholePropagation(in,out);
                }
                else if(dist2(mt) == 1){//eosino
                    int random = distEosi(mt);
                    Mat1f in = Eosi[random];
                    Mat1f out = Mat1f(5,1,0.0f);
                    out.at<float>(1) = 1;
                    convNet.wholePropagation(in,out);
                }
                else if(dist2(mt) == 2){
                    int random = distLymp(mt);
                    Mat1f in = Lymp[random];
                    Mat1f out = Mat1f(5,1,0.0f);
                    out.at<float>(2) = 1;
                    convNet.wholePropagation(in,out);
                }
                else if(dist2(mt) == 3){
                    int random = distMono(mt);
                    Mat1f in = Mono[random];
                    Mat1f out = Mat1f(5,1,0.0f);
                    out.at<float>(3) = 1;
                    convNet.wholePropagation(in,out);
                }
                else if(dist2(mt) == 4){
                    int random = distNeut(mt);
                    Mat1f in = Neut[random];
                    Mat1f out = Mat1f(5,1,0.0f);
                    out.at<float>(4) = 1;
                    convNet.wholePropagation(in,out);
                }





                if(n%8 ==0){
                    convNet.applyError();
                }

                if(n%128==0){

                    //turn off training
                    DropoutLayer * dO = dynamic_cast<DropoutLayer *>(convNet.layers[12].get());

                    dO->training = false;

                    float right = 0,wrong = 0;
                    for(int x = 0;x<40;x++){




                        // so the whole dataset is passed
                        int random = x;
                        Mat1f in,out;
                        in = Mat1f(5,1,0.0f);
                        int sol = dist2(mt);

                        switch(sol){
                            case 0:{
                                in = Baso[random];
                            }
                            case 1:{
                                in = Eosi[random];
                            }
                            case 2:{
                                in = Lymp[random];
                            }
                            case 3:{
                                in = Mono[random];
                            }
                            case 4:{
                                in = Neut[random];
                            }
                        }


                        Mat1f res = convNet.use(in);

                        double min;
                        double max;
                        cv::Point min_loc, max_loc;
                        cv::minMaxLoc(res, &min, &max, &min_loc, &max_loc);

                        if (max_loc.y == sol) {
                            right++;
                        } else {
                            wrong++;
                        }

                        //imshow(std::to_string(max_loc.y),vec[random]);
                        //waitKey();

                    }
                    std::cout << right/(wrong+right)*100 << std::endl;
                    percentagefile << n << "," << right/(wrong+right)*100 << "\n";

                    //turn on training
                    dO->training = true;
                }
                if(n%4096==0){

                    if(n!=0){
                        convNet.save(sf+"-"+std::to_string(n));
                    }
                    percentagefile.close();
                    percentagefile.open(pf,std::ofstream::out | std::ofstream::app);

                }
                if(n%16==0){
                    std::cout <<"n:" << n << std::endl;
                }

            }



        }
        else if(c == 'I'){
            //temporary
            std::vector<Mat1f> TBaso;
            int BasoNum = 53;
            TBaso.resize(BasoNum);

            std::vector<Mat1f> TEosi;
            int EosiNum = 39;
            TEosi.resize(EosiNum);


            std::vector<Mat1f> TLymp;
            int LympNum = 52;
            TLymp.resize(LympNum);

            std::vector<Mat1f> TMono;
            int MonoNum = 48;
            TMono.resize(MonoNum);

            std::vector<Mat1f> TNeut;
            int NeutNum = 50;
            TNeut.resize(NeutNum);


            //real
            std::vector<Mat1f> Baso;
            std::vector<Mat1f> Eosi;
            std::vector<Mat1f> Lymp;
            //std::vector<Mat1f> Mixt;
            std::vector<Mat1f> Mono;
            std::vector<Mat1f> Neut;



            for(int b = 1;b<BasoNum+1;b++){
                Mat temp= ((imread("../Resources/LISC Database/Main Dataset/Baso/"+std::to_string(b)+".bmp",IMREAD_GRAYSCALE)));

                resize(temp,temp,Size(180,144));
                //720*576
                temp.convertTo(TBaso[b-1],CV_32FC2);
                TBaso[b-1]= TBaso[b-1].clone()/256;



            }
            for(int e = 1;e<EosiNum+1;e++){
                Mat temp= ((imread("../Resources/LISC Database/Main Dataset/eosi/"+std::to_string(e)+".bmp",IMREAD_GRAYSCALE)));

                resize(temp,temp,Size(180,144));
                //720*576
                temp.convertTo(TEosi[e-1],CV_32FC2);
                TEosi[e-1] = TEosi[e-1].clone()/256;

                //imshow("test",Eosi[e-1]);
                //waitKey();
            }

            for(int b = 1;b<LympNum+1;b++){
                Mat temp= ((imread("../Resources/LISC Database/Main Dataset/lymp/"+std::to_string(b)+".bmp",IMREAD_GRAYSCALE)));

                resize(temp,temp,Size(180,144));
                //720*576
                temp.convertTo(TLymp[b-1],CV_32FC2);
                TLymp[b-1]= TLymp[b-1].clone()/256;



            }
            for(int e = 1;e<MonoNum+1;e++){
                Mat temp= ((imread("../Resources/LISC Database/Main Dataset/mono/"+std::to_string(e)+".bmp",IMREAD_GRAYSCALE)));

                resize(temp,temp,Size(180,144));
                //720*576
                temp.convertTo(TMono[e-1],CV_32FC2);
                TMono[e-1] = TMono[e-1].clone()/256;

                //imshow("test",Mono[e-1]);
                //waitKey();
            }

            for(int b = 1;b<NeutNum+1;b++){
                Mat temp= ((imread("../Resources/LISC Database/Main Dataset/neut/"+std::to_string(b)+".bmp",IMREAD_GRAYSCALE)));

                resize(temp,temp,Size(180,144));
                //720*576
                temp.convertTo(TNeut[b-1],CV_32FC2);
                TNeut[b-1]= TNeut[b-1].clone()/256;

            }



            //flip images
            BasoNum*=4;
            for(int b = 0;b<TBaso.size();b++){
                Baso.push_back(TBaso[b].clone());

                Mat1f temp = TBaso[b].clone();//clone to be safe
                flip(temp,temp,0);
                Baso.push_back(temp.clone());

                temp = TBaso[b].clone();
                flip(temp,temp,+1);
                Baso.push_back(temp.clone());

                temp = TBaso[b].clone();
                flip(temp,temp,-1);
                Baso.push_back(temp.clone());

            }

            EosiNum*=4;
            for(int b = 0;b<TEosi.size();b++){
                Eosi.push_back(TEosi[b].clone());

                Mat1f temp = TEosi[b].clone();//clone to be safe
                flip(temp,temp,0);
                Eosi.push_back(temp.clone());

                temp = TEosi[b].clone();
                flip(temp,temp,+1);
                Eosi.push_back(temp.clone());

                temp = TEosi[b].clone();
                flip(temp,temp,-1);
                Eosi.push_back(temp.clone());

            }


            LympNum*=4;
            for(int b = 0;b<TLymp.size();b++){
                Lymp.push_back(TLymp[b].clone());

                Mat1f temp = TLymp[b].clone();//clone to be safe
                flip(temp,temp,0);
                Lymp.push_back(temp.clone());

                temp = TLymp[b].clone();
                flip(temp,temp,+1);
                Lymp.push_back(temp.clone());

                temp = TLymp[b].clone();
                flip(temp,temp,-1);
                Lymp.push_back(temp.clone());

            }

            MonoNum*=4;
            for(int b = 0;b<TMono.size();b++){
                Mono.push_back(TMono[b].clone());

                Mat1f temp = TMono[b].clone();//clone to be safe
                flip(temp,temp,0);
                Mono.push_back(temp.clone());

                temp = TMono[b].clone();
                flip(temp,temp,+1);
                Mono.push_back(temp.clone());

                temp = TMono[b].clone();
                flip(temp,temp,-1);
                Mono.push_back(temp.clone());

            }

            NeutNum*=4;
            for(int b = 0;b<TNeut.size();b++){
                Neut.push_back(TNeut[b].clone());

                Mat1f temp = TNeut[b].clone();//clone to be safe
                flip(temp,temp,0);
                Neut.push_back(temp.clone());

                temp = TNeut[b].clone();
                flip(temp,temp,+1);
                Neut.push_back(temp.clone());

                temp = TNeut[b].clone();
                flip(temp,temp,-1);
                Neut.push_back(temp.clone());

            }




            //shuffle
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

            shuffle (Eosi.begin(), Eosi.end(), std::default_random_engine(seed));
            shuffle (Baso.begin(), Baso.end(), std::default_random_engine(seed));
            shuffle (Lymp.begin(), Lymp.end(), std::default_random_engine(seed));
            shuffle (Mono.begin(), Mono.end(), std::default_random_engine(seed));
            shuffle (Neut.begin(), Neut.end(), std::default_random_engine(seed));




            ConvolutionalNetwork convNet(180,144,2,0.07f,"MSE");

            ConvolutionalLayer c0(3,4,0.07f);
            convNet.layers.push_back(std::make_unique<ConvolutionalLayer>(c0));



            ReLULayer rl0;
            convNet.layers.push_back(std::make_unique<ReLULayer>(rl0));

            MaxPoolLayer m0(2,2);
            convNet.layers.push_back(std::make_unique<MaxPoolLayer>(m0));

            ConvolutionalLayer c1(3,4,0.07f);
            convNet.layers.push_back(std::make_unique<ConvolutionalLayer>(c1));

            ReLULayer rl1;
            convNet.layers.push_back(std::make_unique<ReLULayer>(rl1));

            MaxPoolLayer m1(2,2);
            convNet.layers.push_back(std::make_unique<MaxPoolLayer>(m1));

            ConvolutionalLayer c2(3,2,0.07f);
            convNet.layers.push_back(std::make_unique<ConvolutionalLayer>(c2));

            ReLULayer rl2;
            convNet.layers.push_back(std::make_unique<ReLULayer>(rl2));


            FullyConnectedLayer f0(2048,44352,0.07f,"Sigmoid");
            convNet.layers.push_back(std::make_unique<FullyConnectedLayer>(f0));

            FullyConnectedLayer f1(512,2048,0.07f,"Sigmoid");
            convNet.layers.push_back(std::make_unique<FullyConnectedLayer>(f1));

            FullyConnectedLayer f2(128,512,0.07f,"Sigmoid");
            convNet.layers.push_back(std::make_unique<FullyConnectedLayer>(f2));


            DropoutLayer dO(4,25,convNet.outputDim());
            convNet.layers.push_back(std::make_unique<DropoutLayer>(dO));


            FullyConnectedLayer f3(32,128,0.07f,"Sigmoid");
            convNet.layers.push_back(std::make_unique<FullyConnectedLayer>(f3));

            FullyConnectedLayer f4(8,32,0.07f,"Sigmoid");
            convNet.layers.push_back(std::make_unique<FullyConnectedLayer>(f4));

            FullyConnectedLayer f5(5,8,0.07f,"Sigmoid");
            convNet.layers.push_back(std::make_unique<FullyConnectedLayer>(f5));


            int samples;
            std::cout << "Number of samples\n";
            std::cin >> samples;

            std::cout << "percentage file name:\n";
            std::string pf;
            std::cin >> pf;
            std::ofstream percentagefile;
            percentagefile.open(pf);

            std::cout << "savefile name:\n";
            std::string sf;
            std::cin >> sf;

            std::uniform_int_distribution<int> dist2(0,4);

            //-40 so i can use 39 for testing overfitting

        }
        else{
            std::cout << c << " doesn't exist" << std::endl;
        }


    }
    else if(usage == "10"){//network builder


        std::cout << "Welcome to the network building wizard!"<< std::endl;
        std::cout << "Press: \n";

        startRetry:

        std::cout << "1: To load a network and adjust it.\n";
        std::cout << "2: To create a new network form scratch.\n";
        //maybe to visualize a network
        int choice;
        std::cin >> choice;

        ConvolutionalNetwork * network;//TODO: stop memory leak by new maybe smart pointer

        std::string filename;
        switch (choice) {
            case 1: {//load
                std::cout << "Loading Network" << std::endl;
                std::cout << "filename:" << std::endl;

                std::cin >> filename;

                network = new ConvolutionalNetwork(filename);

                break;
            }
            case 2: {//create
                std::cout << "Creating Network" << std::endl;
                std::cout << "filename:" << std::endl;
                std::cin >> filename;

                int w, h, numC, lambda;
                std::cout << "input width:" << std::endl;
                std::cin >> w;

                std::cout << "input height" << std::endl;
                std::cin >> h;

                std::cout << "number of classes:" << std::endl;
                std::cin >> numC;

                std::cout << "training factor/lambda:" << std::endl;
                std::cin >> lambda;

                lossRetry:
                std::cout << "loss function:" << std::endl;
                std::cout << "available functions are:" << std::endl;
                std::cout << "\t MSE" << std::endl;
                std::cout << "\t CrossEntropy" << std::endl;
                std::string lossFunction;
                std::cin >> lossFunction;
                if (lossFunction == "CrossEntropy" || lossFunction == "MSE") {

                } else {
                    std::cout << "Loss function |" << lossFunction << "| doesn't exist" << std::endl;
                    std::cout << "try again" << std::endl;
                    goto lossRetry;
                }
                network = new ConvolutionalNetwork(w,h,numC,lambda,lossFunction);



                break;
            }

            default: {
                std::cout << "You have to choose 1 or 2" << std::endl;
                std::cout << "try again" << std::endl;
                goto startRetry;
            }

            //start modifying the network
            std::string action = "";

            std::cout << "You can start modifying the network you just loaded/created" << std::endl;
            std::cout << "type 'help' for the list of commands" << std::endl;

            while(action != "exit"){
                std::cin >> action;
                if(action == "help"){
                    std::cout << "list of all commands:" << std::endl;
                    std::cout << "\t'help' : displays a list of commands" << std::endl;
                    std::cout << "\t'append' : starts wizzard to append a layer" << std::endl;
                    std::cout << "\t'delete' : starts wizzard to delete a layer" << std::endl;
                    std::cout << "\t'insert' : starts wizzard to insert a layer" << std::endl;
                    std::cout << "\t'list' : lists all layers with their attributes" << std::endl;
                }
                if(action == "append"){

                }
            }

        }

        std::cout << "You're a wizard Harry!\n";
    }
    else if(usage == "11"){

        int samples;
        std::cout << "Number of samples\n";
        std::cin >> samples;

        std::cout << "percentage file name:\n";
        std::string pf;
        std::cin >> pf;
        std::ofstream percentagefile;
        percentagefile.open(pf);

        std::cout << "savefile name:\n";
        std::string sf;
        std::cin >> sf;



        ConvolutionalNetwork convNet(120,120,5,0.07f,"MSE");

        ConvolutionalLayer c0(3,4,0.07f);
        convNet.layers.push_back(std::make_unique<ConvolutionalLayer>(c0));

        ReLULayer rl0;
        convNet.layers.push_back(std::make_unique<ReLULayer>(rl0));

        MaxPoolLayer m0(2,2);
        convNet.layers.push_back(std::make_unique<MaxPoolLayer>(m0));

        ConvolutionalLayer c1(3,4,0.07f);
        convNet.layers.push_back(std::make_unique<ConvolutionalLayer>(c1));

        ReLULayer rl1;
        convNet.layers.push_back(std::make_unique<ReLULayer>(rl1));

        MaxPoolLayer m1(2,2);
        convNet.layers.push_back(std::make_unique<MaxPoolLayer>(m1));

        int a,b,c;
        std::tie(a,b,c) = convNet.outputDim();
        std::cout << a << " " << b << " " << c << std::endl;


        DropoutLayer dl1(1,20,convNet.outputDim());
        convNet.layers.push_back(std::make_unique<DropoutLayer>(dl1));
        //23328


        FullyConnectedLayer f0(4096,a*b*c,0.07f,"Sigmoid");
        convNet.layers.push_back(std::make_unique<FullyConnectedLayer>(f0));

        FullyConnectedLayer f05(1024,4096,0.07f,"Sigmoid");
        convNet.layers.push_back(std::make_unique<FullyConnectedLayer>(f05));

        FullyConnectedLayer f1(128,1024,0.07f,"Sigmoid");
        convNet.layers.push_back(std::make_unique<FullyConnectedLayer>(f1));

        FullyConnectedLayer f2(32,128,0.07f,"Sigmoid");
        convNet.layers.push_back(std::make_unique<FullyConnectedLayer>(f2));

        FullyConnectedLayer f3(5,32,0.07f,"Sigmoid");
        convNet.layers.push_back(std::make_unique<FullyConnectedLayer>(f3));

        //std::vector<Mat1f> vec(300);
        //std::vector<int> labels(300);

        std::vector<std::pair<Mat1f,int>> pvec(300);

        //Read data

        // images
        for(int e = 1;e<300+1;e++){

            std::string fileloc = "../Resources/segmentation_WBC-master/Dataset 1/"+ pad_with_0(e,3) +".bmp";

            Mat temp= ((imread(fileloc,IMREAD_GRAYSCALE)));


            //temp.convertTo(vec[e-1],CV_32FC2);
            temp.convertTo(pvec[e-1].first,CV_32FC2);
            //vec[e-1] = vec[e-1].clone()/256;
            pvec[e-1].first = pvec[e-1].first.clone()/256;

        }

        //training  labels

        std::ifstream labelStream("../Resources/segmentation_WBC-master/Class Labels of Dataset 1.csv");
        assert(labelStream.is_open());


        for(int l = 0;l<300;l++){
            int a,b;
            char coma;
            labelStream >> a >> coma >> b;
            //labels[l]=b;
            pvec[l].second = b-1;

        }

        //shuffle
        //TODO:instert random device
        std::random_shuffle(pvec.begin(),pvec.end());

        //train


        std::uniform_int_distribution<int> dist300(0,250);
        std::uniform_int_distribution<int> distTest(251,299);

        for(int s = 0;s<samples;s++){
            int random = dist300(mt);


            Mat1f desiredOutput(5,1,0.0f);
            desiredOutput.at<float>(pvec[random].second) = 1.0f;

            //to filter images with wrong size
            if(pvec[random].first.rows != 120 || pvec[random].first.cols != 120){
                s--;
                continue;
            }

            convNet.wholePropagation(pvec[random].first,desiredOutput);

            if(s%32 == 0){
                convNet.applyError();
                std::cout << s << std::endl;
            }
            if(s%200==0){
                int right = 0,wrong = 0;
                for(int x = 0;x<50;x++){

                    int r = distTest(mt);

                    //to filter images with wrong size
                    if(pvec[r].first.rows != 120 || pvec[r].first.cols != 120){
                        x--;
                        continue;
                    }

                    Mat1f res = convNet.use(pvec[r].first);

                    //std::cout << r << std::endl;
                    //std::cout << res << std::endl;

                    double min;
                    double max;
                    cv::Point min_loc, max_loc;
                    cv::minMaxLoc(res, &min, &max, &min_loc, &max_loc);
                    if(max_loc.y == pvec[r].second){
                        right++;
                    }
                    else{
                        wrong++;
                    }
                }
                std::cout << "right: " << right << " " <<"wrong: " << wrong << std::endl;
                std::cout << ((float)right/((float)wrong+(float)right))*100 << "%" << std::endl;
                percentagefile <<  s << "," <<((float)right/((float)wrong+(float)right))*100  << "\n";


            }
            if(s%1000 == 0){
                convNet.save(sf + std::to_string(s));
                percentagefile.close();
                percentagefile.open(pf,std::ofstream::out | std::ofstream::app);
            }

        }
        convNet.save(sf + "_final");


    }
    else if(usage == "12"){
        int samples;
        std::cout << "Number of samples\n";
        std::cin >> samples;

        std::cout << "percentage file name:\n";
        std::string pf;
        std::cin >> pf;
        std::ofstream percentagefile;
        //to append

        percentagefile.open(pf,std::ofstream::out | std::ofstream::app);

        std::cout << "Load from:\n";
        std::string sf;
        std::cin >> sf;

        ConvolutionalNetwork convNet(sf);

        convNet.lossFunction = "MSE";//TODO:check for overfitting

        //Read data

        std::vector<std::pair<Mat1f,int>> pvec(300);

        //Read data

        // images
        for(int e = 1;e<300+1;e++){

            std::string fileloc = "../Resources/segmentation_WBC-master/Dataset 1/"+ pad_with_0(e,3) +".bmp";

            Mat temp= ((imread(fileloc,IMREAD_GRAYSCALE)));


            //temp.convertTo(vec[e-1],CV_32FC2);
            temp.convertTo(pvec[e-1].first,CV_32FC2);
            //vec[e-1] = vec[e-1].clone()/256;
            pvec[e-1].first = pvec[e-1].first.clone()/256;

        }

        //training  labels

        std::ifstream labelStream("../Resources/segmentation_WBC-master/Class Labels of Dataset 1.csv");
        assert(labelStream.is_open());


        for(int l = 0;l<300;l++){
            int a,b;
            char coma;
            labelStream >> a >> coma >> b;
            //labels[l]=b;
            pvec[l].second = b-1;

        }

        std::random_shuffle(pvec.begin(),pvec.end());
        //train
        std::uniform_int_distribution<int> dist300(50,299);
        std::uniform_int_distribution<int> distTest(0,49);

        for(int s = 0;s<samples;s++){
            int random = dist300(mt);


            Mat1f desiredOutput(5,1,0.0f);
            desiredOutput.at<float>(pvec[random].second) = 1.0f;

            //to filter images with wrong size
            if(pvec[random].first.rows != 120 || pvec[random].first.cols != 120){
                s--;
                continue;
            }

            convNet.wholePropagation(pvec[random].first,desiredOutput);

            if(s%32 == 0){
                convNet.applyError();
                std::cout << s << std::endl;
            }
            if(s%100==0){
                int right = 0,wrong = 0;
                for(int x = 0;x<50;x++){

                    int r = distTest(mt);

                    //to filter images with wrong size
                    if(pvec[r].first.rows != 120 || pvec[r].first.cols != 120){
                        x--;
                        continue;
                    }

                    Mat1f res = convNet.use(pvec[r].first);

                    std::cout << r << std::endl;
                    std::cout << res << std::endl;

                    double min;
                    double max;
                    cv::Point min_loc, max_loc;
                    cv::minMaxLoc(res, &min, &max, &min_loc, &max_loc);
                    if(max_loc.y == pvec[r].second){
                        right++;
                    }
                    else{
                        wrong++;
                    }
                }
                std::cout << "right: " << right << " " <<"wrong: " << wrong << std::endl;
                std::cout << ((float)right/((float)wrong+(float)right))*100 << "%" << std::endl;
                percentagefile <<  s << "," <<((float)right/((float)wrong+(float)right))*100  << "\n";


            }
            if(s%1000 == 0){
                convNet.save(sf + std::to_string(s));
                percentagefile.close();
                percentagefile.open(pf,std::ofstream::out | std::ofstream::app);
            }

        }
        convNet.save(sf + "_final");
    }
    else if(usage == "13"){
        std::cout << "Load from:\n";
        std::string sf;
        std::cin >> sf;




        ConvolutionalNetwork convNet(sf);

        convNet.lossFunction = "MSE";//TODO:check for overfitting

        std::vector<Mat1f> vec(300);
        std::vector<int> labels(300);

        //Read data

        //  images
        for(int e = 1;e<300+1;e++){

            std::string fileloc = "../Resources/segmentation_WBC-master/Dataset 1/"+ pad_with_0(e,3) +".bmp";

            Mat temp= ((imread(fileloc,IMREAD_GRAYSCALE)));


            temp.convertTo(vec[e-1],CV_32FC2);
            vec[e-1] = vec[e-1].clone()/256;


        }

        //  labels

        std::ifstream labelStream("../Resources/segmentation_WBC-master/Class Labels of Dataset 1.csv");
        assert(labelStream.is_open());


        for(int l = 0;l<300;l++){
            int a,b;
            char coma;
            labelStream >> a >> coma >> b;
            labels[l]=b-1;

        }

        std::cout << "Enter" << std::endl;
        std::cout << "1: Statistic output" << " 2: Images" <<" 3: Statistics on different dataset" <<" 4 :Images of different dataset" << " 5:Statistic on mirrored images"<< std::endl;
        std::string us;
        std::cin >> us;

        std::cout << "Number of samples" << std::endl;
        int samples;
        std::cin >> samples;



        std::uniform_int_distribution<int> dist300(0,299);

        //use
        if(us == "1"){//statistics
            float r = 0,w = 0;
            for(int s = 0;s<samples;s++) {
                int random = dist300(mt);

                Mat1f desiredOutput(5, 1, 0.0f);
                desiredOutput.at<float>(labels[random]) = 1.0f;

                //to filter images with wrong size
                if (vec[random].rows != 120 || vec[random].cols != 120) {
                    s--;
                    continue;
                }
                double min;
                double max;
                cv::Point min_loc, max_loc;
                cv::minMaxLoc(convNet.use(vec[random]), &min, &max, &min_loc, &max_loc);
                if(max_loc.y == labels[random]){
                    r++;
                }
                else{
                    w++;
                }



            }

            std::cout << (r/(r+w))*100 << "%" << std::endl;
            std::cout << r << " " << w << std::endl;
        }
        else if(us == "2"){//images
            std::vector<std::string> labelname={"neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil"};
            for(int s = 0;s<samples;s++) {
                int random = dist300(mt);




                //to filter images with wrong size
                if (vec[random].rows != 120 || vec[random].cols != 120) {
                    s--;
                    continue;
                }

                double min;
                double max;
                cv::Point min_loc, max_loc;
                cv::minMaxLoc(convNet.use(vec[random]), &min, &max, &min_loc, &max_loc);

                std::cout << "Network predicted: " <<  labelname[max_loc.y]<< " correct solution was: "<< labelname[labels[random]] <<std::endl;
                imshow("Image",vec[random]);
                waitKey();

            }
        }
        else if(us == "3"){
            std::ifstream LS("../Resources/segmentation_WBC-master/Class Labels of Dataset 2.csv");
            assert(LS.is_open());
            std::vector<int> l(100);
            std::vector<Mat1f> v(100);

            std::uniform_int_distribution<int> dist100(0,99);

            for(int la = 0;la<100;la++){
                int a,b;
                char coma;
                LS >> a >> coma >> b;
                l[la]=b;

            }

            for(int e = 1;e<100+1;e++){

                std::string fileloc = "../Resources/segmentation_WBC-master/Dataset 2/"+ pad_with_0(e,3) +".bmp";

                Mat temp= ((imread(fileloc,IMREAD_GRAYSCALE)));

                Size size(120,120);
                resize(temp,temp,size);

                temp.convertTo(v[e-1],CV_32FC2);
                v[e-1] = v[e-1].clone()/256;


            }

            float r = 0,w = 0;
            for(int s = 0;s<samples;s++) {
                int random = dist100(mt);

                Mat1f desiredOutput(5, 1, 0.0f);
                desiredOutput.at<float>(l[random]) = 1.0f;

                //to filter images with wrong size
                if (v[random].rows != 120 || v[random].cols != 120) {
                    s--;
                    continue;
                }
                double min;
                double max;
                cv::Point min_loc, max_loc;
                cv::minMaxLoc(convNet.use(v[random]), &min, &max, &min_loc, &max_loc);
                if(max_loc.y == l[random]){
                    r++;
                }
                else{
                    w++;
                }



            }

            std::cout << (r/(r+w))*100 << "%" << std::endl;
            std::cout << r << " " << w << std::endl;



        }
        else if(us == "4"){
            std::ifstream LS("../Resources/segmentation_WBC-master/Class Labels of Dataset 2.csv");
            assert(LS.is_open());
            std::vector<int> l(100);
            std::vector<Mat1f> v(100);

            std::uniform_int_distribution<int> dist100(0,99);



            for(int la = 0;la<100;la++){
                int a,b;
                char coma;
                LS >> a >> coma >> b;
                std::cout << a << b;
                l[la]=b;

            }

            for(int e = 1;e<100+1;e++){

                std::string fileloc = "../Resources/segmentation_WBC-master/Dataset 2/"+ pad_with_0(e,3) +".bmp";

                Mat temp= ((imread(fileloc,IMREAD_GRAYSCALE)));

                Size size(120,120);
                resize(temp,temp,size);

                temp.convertTo(v[e-1],CV_32FC2);
                v[e-1] = v[e-1].clone()/256;


            }
            std::vector<std::string> labelname={"neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil"};
            for(int s = 0;s<samples;s++) {

                int random = dist100(mt);
                std::cout << "random " << random << std::endl;

                double min;
                double max;
                cv::Point min_loc, max_loc;
                cv::minMaxLoc(convNet.use(v[random]), &min, &max, &min_loc, &max_loc);

                std::cout << "Network predicted: " << labelname[max_loc.y - 1] << " correct solution was: " << labelname[l[random] - 1] << std::endl;
                imshow("Image", v[random]);
                waitKey();
            }

        }
        else if(us == "5"){
            float r = 0,w = 0;
            for(int v = 0;v<vec.size();v++){
                flip(vec[v],vec[v],-1);
            }
            for(int s = 0;s<samples;s++) {
                int random = dist300(mt);

                Mat1f desiredOutput(5, 1, 0.0f);
                desiredOutput.at<float>(labels[random]) = 1.0f;

                //to filter images with wrong size
                if (vec[random].rows != 120 || vec[random].cols != 120) {
                    s--;
                    continue;
                }
                double min;
                double max;
                cv::Point min_loc, max_loc;
                cv::minMaxLoc(convNet.use(vec[random]), &min, &max, &min_loc, &max_loc);
                if(max_loc.y == labels[random]){
                    r++;
                }
                else{
                    w++;
                }



            }

            std::cout << (r/(r+w))*100 << "%" << std::endl;
            std::cout << r << " " << w << std::endl;
        }


    }
    else if(usage == "14"){
        std::string data = "../Resources/BloodCells/train.csv";
        std::ifstream dataIn(data);
        //12445
        std::vector<Mat1f> E;
        std::vector<Mat1f> L;
        std::vector<Mat1f> M;
        std::vector<Mat1f> N;

        std::string aa,bb;
        dataIn >> aa >> bb;

        for(int l = 1;l<12437;l++){
            std::string filepath,Type;
            dataIn >> filepath >> Type;

            Mat temp = imread(filepath,IMREAD_GRAYSCALE);

            resize(temp,temp,Size(80,60));
            temp.convertTo(temp,CV_32FC2);
            temp/=256;



            if(Type[0] == 'E'){
                E.push_back(temp.clone());

            }
            if(Type[0] == 'L'){
                L.push_back(temp.clone());

            }
            if(Type[0] == 'M'){
                M.push_back(temp.clone());

            }
            if(Type[0] == 'N'){
                N.push_back(temp.clone());

            }


        }
        ConvolutionalNetwork convNet(80,60,4,0.07f,"MSE");

        ConvolutionalLayer c0(3,4,0.07f);
        convNet.layers.push_back(std::make_unique<ConvolutionalLayer>(c0));

        ReLULayer rl0;
        convNet.layers.push_back(std::make_unique<ReLULayer>(rl0));

        ConvolutionalLayer c1(3,4,0.07f);
        convNet.layers.push_back(std::make_unique<ConvolutionalLayer>(c1));

        ReLULayer rl1;
        convNet.layers.push_back(std::make_unique<ReLULayer>(rl1));

        MaxPoolLayer mp0(2,2);
        convNet.layers.push_back(std::make_unique<MaxPoolLayer>(mp0));


        ConvolutionalLayer c2(3,2,0.07f);
        convNet.layers.push_back(std::make_unique<ConvolutionalLayer>(c2));

        ReLULayer rl2;
        convNet.layers.push_back(std::make_unique<ReLULayer>(rl2));

        ConvolutionalLayer c3(3,2,0.07f);
        convNet.layers.push_back(std::make_unique<ConvolutionalLayer>(c3));

        ReLULayer rl3;
        convNet.layers.push_back(std::make_unique<ReLULayer>(rl3));

        MaxPoolLayer mp1(2,2);
        convNet.layers.push_back(std::make_unique<MaxPoolLayer>(mp1));


        int a,b,c;
        std::tie(a,b,c) = convNet.outputDim();
        std::cout << a << " " << b << " " << c << std::endl;


        /*
        FullyConnectedLayer fc(4112,a*b*c,0.07f,"Sigmoid");
        convNet.layers.push_back(std::make_unique<FullyConnectedLayer>(fc));

         */
        FullyConnectedLayer fc0(128,a*b*c,0.07f,"Sigmoid");
        convNet.layers.push_back(std::make_unique<FullyConnectedLayer>(fc0));

        FullyConnectedLayer fc1(128,128,0.07f,"Sigmoid");
        convNet.layers.push_back(std::make_unique<FullyConnectedLayer>(fc1));

        DropoutLayer dO(4,25,convNet.outputDim());
        convNet.layers.push_back(std::make_unique<DropoutLayer>(dO));

        FullyConnectedLayer fc2(4,128,0.07f,"Sigmoid");
        convNet.layers.push_back(std::make_unique<FullyConnectedLayer>(fc2));




        std::uniform_int_distribution<int> distEosi(40,E.size()-1);
        std::uniform_int_distribution<int> distLymp(40,L.size()-1);
        std::uniform_int_distribution<int> distMono(40,M.size()-1);
        std::uniform_int_distribution<int> distNeut(40,N.size()-1);


        std::uniform_int_distribution<int> distTest(0,39);
        std::uniform_int_distribution<int> dist2(1,4);

        std::cout << "pf" << std::endl;
        std::string pf;
        std::cin >> pf;
        std::ofstream percentagefile(pf);

        std::cout << "sf" << std::endl;
        std::string sf;
        std::cin >> sf;




        std::cout << "samples" << std::endl;
        float samples;
        std::cin >> samples;
        std::cout << "This is equal to " << samples/12437 << " epochs" << std::endl;

        for(int n = 0;n<samples;n++){

            if(dist2(mt) == 1){//eosino
                int random = distEosi(mt);
                Mat1f in = E[random];
                Mat1f out = Mat1f(4,1,0.0f);
                out.at<float>(0) = 1;
                convNet.wholePropagation(in,out);
            }
            else if(dist2(mt) == 2){
                int random = distLymp(mt);
                Mat1f in = L[random];
                Mat1f out = Mat1f(4,1,0.0f);
                out.at<float>(1) = 1;
                convNet.wholePropagation(in,out);
            }
            else if(dist2(mt) == 3){
                int random = distMono(mt);
                Mat1f in = M[random];
                Mat1f out = Mat1f(4,1,0.0f);
                out.at<float>(2) = 1;
                convNet.wholePropagation(in,out);
            }
            else if(dist2(mt) == 4){
                int random = distNeut(mt);
                Mat1f in = N[random];
                Mat1f out = Mat1f(4,1,0.0f);
                out.at<float>(3) = 1;
                convNet.wholePropagation(in,out);
            }





            if(n%8 ==0){
                convNet.applyError();
            }

            if(n%128==0){

                //turn off training
                DropoutLayer * dO = dynamic_cast<DropoutLayer *>(convNet.layers[12].get());

                dO->training = false;

                float right = 0,wrong = 0;
                for(int x = 0;x<32;x++){



                    int random = distTest(mt);
                    Mat1f in,out;
                    in = Mat1f(4,1,0.0f);
                    int sol = dist2(mt);

                    switch(sol){
                        case 1:{
                            in = E[random];
                        }
                        case 2:{
                            in = L[random];
                        }
                        case 3:{
                            in = M[random];
                        }
                        case 4:{
                            in = N[random];
                        }
                    }

                    sol--;


                    Mat1f res = convNet.use(in);

                    double min;
                    double max;
                    cv::Point min_loc, max_loc;
                    cv::minMaxLoc(res, &min, &max, &min_loc, &max_loc);

                    if (max_loc.y == sol) {
                        right++;
                    } else {
                        wrong++;
                    }

                    //std::cout << res << std::endl;
                    //std::cout << max_loc.y << " " << sol << std::endl;
                    //imshow(std::to_string(max_loc.y),vec[random]);
                    //waitKey();

                }
                std::cout << (right/(wrong+right)*100)-25 << std::endl;
                percentagefile << n << "," << right/(wrong+right)*100 << "\n";

                //turn on training
                dO->training = true;
            }
            if(n%4096==0){

                if(n!=0){
                    convNet.save(sf+"-"+std::to_string(n));
                }
                percentagefile.close();
                percentagefile.open(pf,std::ofstream::out | std::ofstream::app);

            }
            if(n%16==0){
                std::cout <<"n:" << n << std::endl;
            }

        }




    }
    else if(usage == "15"){

        std::cout << "WBC networkfile" << std::endl;
        std::string nf;
        std::cin >> nf;

        ConvolutionalNetwork convNet(nf);
        convNet.lossFunction = "MSE";


        std::cout << "File" << std::endl;
        std::string file;
        std::cin >> file;
        file = "../Resources/segmentation_WBC-master/Dataset 2/001.bmp";
        Mat1f image = imread(file,IMREAD_GRAYSCALE);
        Mat1f TempImage;
        image/=256;

        unsigned int x,y,w,h;
        x = 0;
        y = 0;
        w = 40;
        h = 40;

        namedWindow("Live example");



        while(true){
            Rect r(x+1,y+1,w+1,h+1);
            TempImage = image.clone();
            rectangle(TempImage,r,(255,0,0));

            imshow("Live example",TempImage);


            int key = waitKey();
            if(key == 'w'){
                y--;
            }
            else if (key == 's'){
                y++;
            }
            else if(key == 'a'){
                x--;
            }
            else if(key == 'd'){
                x++;
            }
            else if(key == 'q'){//smaller
                w--;
                h--;
            }
            else if(key == 'e'){//bigger
                w++;
                h++;
            }
            else if(key == 27){//key esc
                return 0;
            }
            else if(key == 'r'){//result
                Mat1f input(image,r);
                resize(input,input,Size(120,120));

                Mat1f output = convNet.use(input);

                double min,max;
                Point minPos,maxPos;

                minMaxLoc(output,&min,&max,&minPos,&maxPos);
                std::cout << maxPos.y << std::endl;
            }
            else{
                std::cout << key << std::endl;
            }


        }




    }
    else{
        std::cout << "Sorry but: " << usage << " doesn't exist" << std::endl;



    }





    return 0;
}