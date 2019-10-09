#ifndef NN0_1_GENE_H
#define NN0_1_GENE_H


#include <vector>
#include <string>
#include <cassert>
#include <cmath>

class gene {//TODO:remove ugly rand()
public:

    int randRangeConv = 10;
    int randRangeFC = 10000;
    int randRangeMP = 10;


    int lenght;
    int numClasses;
    std::vector<std::string> alleles;
    int exclusion;
    gene(int lenght,int numClasses,int exclusion);
    gene(gene p1,gene p2);
    gene randomize(float probabilityL,float probability);


    std::string generateRandomAllele(bool D2);

};


#endif //NN0_1_GENE_H
