#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

struct Node {
    float input;
    float output;
    float bias;
    float delta;

    float sigmoid(float x) {
        return 1.0 / (1.0 + exp(-x));
    }

    float sigmoid_derivative(float x) {
        float sig = sigmoid(x);
        return sig * (1.0 - sig);
    }
};

struct Edge {
    Node* from;
    Node* to;
    float weight;
    float weight_gradient;

    void set_nodes(Node* from_node, Node* to_node) {
        from = from_node;
        to = to_node;
    }
};

void accumulate_gradients(Node* input_layer, Node* layer2, Node* layer3, Node* output_layer, Edge edges1[2][16], Edge edges2[16][4], Edge edges3[4], float* targets) {
    // Calculate delta for output_layer
    float error = targets[0] - output_layer[0].output;
    output_layer[0].delta = error * output_layer[0].sigmoid_derivative(output_layer[0].output);

    // Accumulate gradients for edges between layer3 and output_layer
    for (int i = 0; i < 4; ++i) {
        float gradient = layer3[i].output * output_layer[0].delta;
        edges3[i].weight_gradient += gradient;
    }

    // Calculate deltas for layer3
    for (int i = 0; i < 4; ++i) {
        float error = output_layer[0].delta * edges3[i].weight;
        layer3[i].delta = error * layer3[i].sigmoid_derivative(layer3[i].output);
    }

    // Accumulate gradients for edges between layer2 and layer3
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 4; ++j) {
            float gradient = layer2[i].output * layer3[j].delta;
            edges2[i][j].weight_gradient += gradient;
        }
    }

    // Calculate deltas for layer2
    for (int i = 0; i < 16; ++i) {
        float error = 0.0;
        for (int j = 0; j < 4; ++j) {
            error += layer3[j].delta * edges2[i][j].weight;
        }
        layer2[i].delta = error * layer2[i].sigmoid_derivative(layer2[i].output);
    }

    // Accumulate gradients for edges between input_layer and layer2
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 16; ++j) {
            float gradient = input_layer[i].output * layer2[j].delta;
            edges1[i][j].weight_gradient += gradient;
        }
    }
}

void update_weights_and_biases(Node* layer2, Node* layer3, Node* output_layer, Edge edges1[2][16], Edge edges2[16][4], Edge edges3[4], float learning_rate) {
    // Update weights and biases for edges between layer3 and output_layer
    for (int i = 0; i < 4; ++i) {
        edges3[i].weight += learning_rate * edges3[i].weight_gradient;
        edges3[i].weight_gradient = 0.0; // Reset gradient
        layer3[i].bias += learning_rate * layer3[i].delta;
    }

    // Update weights and biases for edges between layer2 and layer3
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 4; ++j) {
            edges2[i][j].weight += learning_rate * edges2[i][j].weight_gradient;
            edges2[i][j].weight_gradient = 0.0; // Reset gradient
        }
        layer2[i].bias += learning_rate * layer2[i].delta;
    }

    // Update weights and biases for edges between input_layer and layer2
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 16; ++j) {
            edges1[i][j].weight += learning_rate * edges1[i][j].weight_gradient;
            edges1[i][j].weight_gradient = 0.0; // Reset gradient
        }
    }

    // Update bias for output_layer
    output_layer[0].bias += learning_rate * output_layer[0].delta;

    // Print weights for debugging
//    std::cout << "Weights after update:" << std::endl;
//    for (int i = 0; i < 2; ++i) {
//        for (int j = 0; j < 16; ++j) {
//            std::cout << "edges1[" << i << "][" << j << "].weight: " << edges1[i][j].weight << std::endl;
//        }
//    }
//    for (int i = 0; i < 16; ++i) {
//        for (int j = 0; j < 4; ++j) {
//            std::cout << "edges2[" << i << "][" << j << "].weight: " << edges2[i][j].weight << std::endl;
//        }
//    }
//    for (int i = 0; i < 4; ++i) {
//        std::cout << "edges3[" << i << "].weight: " << edges3[i].weight << std::endl;
//    }
}

int main() {
    srand(time(NULL));

    Node input_layer[2];
    Node layer2[16];
    Node layer3[4];
    Node output_layer[1];

    Edge edges1[2][16];
    Edge edges2[16][4];
    Edge edges3[4];

    // Initialize edges between input_layer and layer2
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 16; ++j) {
            edges1[i][j].set_nodes(&input_layer[i], &layer2[j]);
            edges1[i][j].weight = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f;
        }
    }

    // Initialize edges between layer2 and layer3
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 4; ++j) {
            edges2[i][j].set_nodes(&layer2[i], &layer3[j]);
            edges2[i][j].weight = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f;
        }
    }

    // Initialize edges between layer3 and output_layer
    for (int i = 0; i < 4; ++i) {
        edges3[i].set_nodes(&layer3[i], &output_layer[0]);
        edges3[i].weight = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f;
    }

    // Example training data
    float inputs[4][2] = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    float targets[4][1] = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };
    float learning_rate = 0.01;

    // Training loop
    for (int epoch = 0; epoch < 10000000; ++epoch) {
        float total_loss = 0.0;

        for (int example = 0; example < 4; ++example) {
            // Set input values for input_layer
            for (int i = 0; i < 2; ++i) {
                input_layer[i].output = inputs[example][i];
            }

            // Forward pass
            // Layer 1 to Layer 2
            for (int i = 0; i < 16; ++i) {
                layer2[i].input = 0.0;
                for (int j = 0; j < 2; ++j) {
                    layer2[i].input += input_layer[j].output * edges1[j][i].weight;
                }
                layer2[i].output = layer2[i].sigmoid(layer2[i].input + layer2[i].bias);
            }

            // Layer 2 to Layer 3
            for (int i = 0; i < 4; ++i) {
                layer3[i].input = 0.0;
                for (int j = 0; j < 16; ++j) {
                    layer3[i].input += layer2[j].output * edges2[j][i].weight;
                }
                layer3[i].output = layer3[i].sigmoid(layer3[i].input + layer3[i].bias);
            }

            // Layer 3 to Output Layer
            output_layer[0].input = 0.0;
            for (int i = 0; i < 4; ++i) {
                output_layer[0].input += layer3[i].output * edges3[i].weight;
            }
            output_layer[0].output = output_layer[0].sigmoid(output_layer[0].input + output_layer[0].bias);

            // Calculate loss
            float error = targets[example][0] - output_layer[0].output;
            total_loss += error * error;

            // Backpropagation
            accumulate_gradients(input_layer, layer2, layer3, output_layer, edges1, edges2, edges3, targets[example]);

            // Update weights and biases
            update_weights_and_biases(layer2, layer3, output_layer, edges1, edges2, edges3, learning_rate);
        }

        // Optionally print loss every few epochs to monitor convergence
        if (epoch % 10000 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss << std::endl;
        }
    }
    // Print the output of the input data after training
    std::cout << "Outputs after training:" << std::endl;
    for (int example = 0; example < 4; ++example) {
        // Set input values for input_layer
        for (int i = 0; i < 2; ++i) {
            input_layer[i].output = inputs[example][i];
        }

        // Forward pass
        // Layer 1 to Layer 2
        for (int i = 0; i < 16; ++i) {
            layer2[i].input = 0.0;
            for (int j = 0; j < 2; ++j) {
                layer2[i].input += input_layer[j].output * edges1[j][i].weight;
            }
            layer2[i].output = layer2[i].sigmoid(layer2[i].input + layer2[i].bias);
        }

        // Layer 2 to Layer 3
        for (int i = 0; i < 4; ++i) {
            layer3[i].input = 0.0;
            for (int j = 0; j < 16; ++j) {
                layer3[i].input += layer2[j].output * edges2[j][i].weight;
            }
            layer3[i].output = layer3[i].sigmoid(layer3[i].input + layer3[i].bias);
        }

        // Layer 3 to Output Layer
        output_layer[0].input = 0.0;
        for (int i = 0; i < 4; ++i) {
            output_layer[0].input += layer3[i].output * edges3[i].weight;
        }
        output_layer[0].output = output_layer[0].sigmoid(output_layer[0].input + output_layer[0].bias);

        // Print the output
        std::cout << "Input: {" << inputs[example][0] << ", " << inputs[example][1] << "} -> Output: " << output_layer[0].output << std::endl;
    }
    return 0;
}
