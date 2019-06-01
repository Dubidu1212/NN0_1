#include <iostream>
#include "FeedForwardNetwork.h"


int main() {
    FeedForwardNetwork ffn({2,3,3,2,1});
    ffn.print();


    return 0;
}