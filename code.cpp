#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <iomanip> 
using namespace std;

struct Connection {
    double weight;
    double deltaWeight;
};

class Neuron;
typedef vector<Neuron> Layer;

class Neuron {
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val) { m_outputVal = val; }
    double getOutputVal() const { return m_outputVal; }
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

private:
    static double eta;
    static double alpha;
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight() { return rand() / double(RAND_MAX); }
    double sumDOW(const Layer &nextLayer) const;
    double m_outputVal;
    vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
};

double Neuron::eta = 0.15;
double Neuron::alpha = 0.15;

void Neuron::updateInputWeights(Layer &prevLayer) {
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
        double newDeltaWeight =
            eta * neuron.getOutputVal() * m_gradient + alpha * oldDeltaWeight;
        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

double Neuron::sumDOW(const Layer &nextLayer) const {
    double sum = 0.0;
    for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }
    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer) {
    double dow = sumDOW(nextLayer);
    m_gradient = dow * transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal) {
    double delta = targetVal - m_outputVal;
    m_gradient = delta * transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x) {
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x) {
    return 1.0 - x * x;
}

void Neuron::feedForward(const Layer &prevLayer) {
    double sum = 0.0;
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() *
               prevLayer[n].m_outputWeights[m_myIndex].weight;
    }
    m_outputVal = transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex) {
    for (unsigned c = 0; c < numOutputs; ++c) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    m_myIndex = myIndex;
}

class Net {
public:
    Net(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultsVals) const;

private:
    vector<Layer> m_Layers;
    double m_error;
    double m_recentAvgError;
    double m_recentAvgSmoothingFactor = 100.0;
};

void Net::getResults(vector<double> &resultsVals) const {
    resultsVals.clear();
    for (unsigned n = 0; n < m_Layers.back().size() - 1; ++n) {
        resultsVals.push_back(m_Layers.back()[n].getOutputVal());
    }
}

void Net::backProp(const vector<double> &targetVals) {
    Layer &outputLayer = m_Layers.back();
    m_error = 0.0;
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1;
    m_error = sqrt(m_error);
    m_recentAvgError = (m_recentAvgError * m_recentAvgSmoothingFactor + m_error) / (m_recentAvgSmoothingFactor + 1.0);
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }
    for (unsigned layerNum = m_Layers.size() - 2; layerNum > 0; --layerNum) {
        Layer &hiddenLayer = m_Layers[layerNum];
        Layer &nextLayer = m_Layers[layerNum + 1];
        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }
    for (unsigned layerNum = m_Layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = m_Layers[layerNum];
        Layer &prevLayer = m_Layers[layerNum - 1];
        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void Net::feedForward(const vector<double> &inputVals) {
    assert(inputVals.size() == m_Layers[0].size() - 1);
    for (unsigned i = 0; i < inputVals.size(); ++i) {
        m_Layers[0][i].setOutputVal(inputVals[i]);
    }
    for (unsigned layerNum = 1; layerNum < m_Layers.size(); ++layerNum) {
        Layer &prevLayer = m_Layers[layerNum - 1];
        for (unsigned n = 0; n < m_Layers[layerNum].size() - 1; ++n) {
            m_Layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

Net::Net(const vector<unsigned> &topology) {
    for (unsigned layerNum = 0; layerNum < topology.size(); ++layerNum) {
        m_Layers.push_back(Layer());
        unsigned numOutputs = (layerNum == topology.size() - 1) ? 0 : topology[layerNum + 1];
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            m_Layers.back().push_back(Neuron(numOutputs, neuronNum));
        }
    }
}




using namespace std;

int main() {
    vector<unsigned> topology = {3, 2, 1}; 
    Net myNet(topology);

    // Training dataset (XOR-like problem)
    vector<vector<double>> trainingInputs = {
        {0.0, 0.0, 1.0}, // Example input 1
        {0.0, 1.0, 1.0}, // Example input 2
        {1.0, 0.0, 1.0}, // Example input 3
        {1.0, 1.0, 1.0}  // Example input 4
    };
    
    vector<vector<double>> trainingTargets = {
        {0.0},  // Expected output for input 1
        {1.0},  // Expected output for input 2
        {1.0},  // Expected output for input 3
        {0.0}   // Expected output for input 4
    };

    const int trainingEpochs = 10000; // Number of times we train

    cout << "Training the network...\n";

    for (int i = 0; i < trainingEpochs; ++i) {
        double totalError = 0.0;
        
        for (size_t j = 0; j < trainingInputs.size(); ++j) {
            myNet.feedForward(trainingInputs[j]);
            myNet.backProp(trainingTargets[j]);

            vector<double> resultsVals;
            myNet.getResults(resultsVals);

            double error = pow(resultsVals[0] - trainingTargets[j][0], 2);
            totalError += error;
        }

        if (i % 1000 == 0) { // Print error every 1000 epochs
            cout << "Epoch " << i << " - Error: " << totalError / trainingInputs.size() << endl;
        }
    }

    // Testing after training
    cout << "\nTesting the trained network:\n";
    int correctPredictions = 0;
    for (size_t j = 0; j < trainingInputs.size(); ++j) {
        myNet.feedForward(trainingInputs[j]);
        vector<double> resultsVals;
        myNet.getResults(resultsVals);

        double predictedOutput = (resultsVals[0] >= 0.5) ? 1.0 : 0.0;
        double expectedOutput = trainingTargets[j][0];

        cout << fixed << setprecision(3);
        cout << "Input: [" << trainingInputs[j][0] << ", " << trainingInputs[j][1] << ", " << trainingInputs[j][2] << "] ";
        cout << "Predicted: " << resultsVals[0] << " (Rounded: " << predictedOutput << ") ";
        cout << "Expected: " << expectedOutput << endl;

        if (predictedOutput == expectedOutput) correctPredictions++;
    }

    double accuracy = (correctPredictions / (double)trainingInputs.size()) * 100;
    cout << "\nFinal Accuracy: " << accuracy << "%\n";

    return 0;
}
