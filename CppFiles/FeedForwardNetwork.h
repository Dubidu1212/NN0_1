#ifndef NN0_1_FEEDFORWARDNETWORK_H
#define NN0_1_FEEDFORWARDNETWORK_H

#include <opencv2/opencv.hpp>
#include "helperFunctions.h"

using namespace cv;
//! Class for standard Feed Forward Neural Network
/*!
 * This class uses backpropagation to train a function to match input vectors to output vectors.
 * Input vectors can be anything from images to true or false statements.
 * Input has to be normalized to be between 0 and 1.
 */

class FeedForwardNetwork{
private:
    //! Container of all biases of the Neural Network.
    /*! Vector contains int layers-1 Biasmatrices.
     *  Matrix in layer i has dimensions \e layerTemplate[i] * 1
     */
    std::vector<Mat1f> b;


    //! Container of all weights of the Neural Network.
    /*! Vector contains int layers Weightmatrices
     *  Matrix in layer i has dimensions \e layerTemplate[i+1] * \e layerTemplate[i]
     */
    std::vector<Mat1f> w;


    //! Contains for each layer its size.
    std::vector<int> layerTemplate;



    //! Number of layers of network.
    int layers;
public:

    //! Default constructor of Feed Forward Network
    /*! @param layersizes a vector which indicates the number of nodes per layer, 0 indexed.
     *
     */
    FeedForwardNetwork(std::vector<int> layersizes);

    //! Function to use a trained network.
    /*! Takes as input a vector of predefined size and outputs another vector of a different also predetermined size.*/
    Mat1f use(Mat1f in);

    //! Function to train the network in batches.
    /*! Takes as input a vector of pairs, which are a input and asociated output.*/
    void trainBatch(std::vector<std::pair<Mat1f,Mat1f>> batch);

    //! Function that prints the weights and biases of the network.
    void print();

};

#endif //NN0_1_FEEDFORWARDNETWORK_H