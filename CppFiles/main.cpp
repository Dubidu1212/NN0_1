#include <iostream>
#include "FeedForwardNetwork.h"


int main() {
    bool FeedForward = false;
    if(FeedForward){
        FeedForwardNetwork ffn({2,10,10,2},0.1);
        //ffn.print();
        std::cout << "0,0 " << ffn.use(Mat1f(2,1,0.0)) << std::endl;
        std::cout << "1,1 " << ffn.use(Mat1f(2,1,1.0)) << std::endl;
        std::cout << "1,0 " << ffn.use((Mat1f(2,1) << 1,0)) << std::endl;
        std::cout << "0,1 " << ffn.use((Mat1f(2,1) << 0,1)) << std::endl;
        for(int iteration = 0;iteration<10;iteration++){
            int a,b;
            a = rand()%2;
            b = rand()%2;

            int out;
            if(a == b && a == 0){
                out = 0;
                ffn.trainSingle((Mat1f(2,1) << a,b),(Mat1f(2,1)<< 0,1));
            }
            else{
                ffn.trainSingle((Mat1f(2,1) << a,b),(Mat1f(2,1)<< 1,0));
            }
            std::cout << a << " " << b << " " << out << std::endl;

        }
        //ffn.print();
        std::cout << "0,0 \n" << ffn.use(Mat1f(2,1,0.0)) << std::endl;
        std::cout << "1,1 \n" << ffn.use(Mat1f(2,1,1.0)) << std::endl;
        std::cout << "1,0 \n" << ffn.use((Mat1f(2,1) << 1,0)) << std::endl;
        std::cout << "0,1 \n" << ffn.use((Mat1f(2,1) << 0,1)) << std::endl;
    }
    else{

    }





    return 0;
}