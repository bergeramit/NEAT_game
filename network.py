import numpy as np
import random

'''
Values {node_index: value}
weights {(5,1): 4, (4,1): 5}
for 5 -> value += weights[edge] * values[edge[0]]
'''

# x, y distance from goal
INPUT_NODES_NUM = 3
# up down left right
OUTPUT_NODES_NUMBER = 4
BIAS_VALUE = -0.5
DIRECTIONS_TO_PLAY = {0: 'down',1: 'up',2: 'left',3: 'right'}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calculate_direction(output):
    '''
    calculate the direction to got based on the max element in the array
    '''
    max = -100
    direction = -1
    for i in range(OUTPUT_NODES_NUMBER):
        if max < output[i, 0]:
            max = output[i, 0]
            direction = i
    print("-----------------------")
    print(output)
    print("Move To make: " + DIRECTIONS_TO_PLAY[direction])
    print("-----------------------")
    return DIRECTIONS_TO_PLAY[direction]


class NeuralNetwork:
    def __init__(self):
        self.neuron_count = INPUT_NODES_NUM + 1
        self.input_neurons = np.ones((1, INPUT_NODES_NUM), dtype=np.float32)
        self.bias = BIAS_VALUE
        self.hidden_neurons = 0
        self.weights_out = np.random.random((OUTPUT_NODES_NUMBER, INPUT_NODES_NUM))
        self.output_neuron = np.zeros((OUTPUT_NODES_NUMBER, 1), dtype=np.float32)
        self.hidden_neurons_weights_in = None
        self.hidden_neurons_weights_out = None


    def __repr__(self):
        reprr = ''
        reprr += "Network: \n"
        reprr += "Input Neurons Count: {}\n".format(INPUT_NODES_NUM)
        reprr += "Input Neurons: \n"
        reprr += str(self.input_neurons) + '\n'
        reprr += "Hidden Neurons Count: {}\n".format(self.hidden_neurons)
        reprr += "--------Weights from input to output---------\n"
        reprr += str(self.weights_out) + '\n'
        reprr += "--------Weights from input to hidden---------\n"
        reprr += str(self.hidden_neurons_weights_in) + '\n'
        reprr += "--------Weights from hidden to output---------\n"
        reprr += str(self.hidden_neurons_weights_out) + '\n'
        reprr += "--------Neuron output---------\n"
        reprr += str(self.output_neuron) + '\n'
        return reprr


    def add_node(self):
        '''
        adding new hidden node to the system
        '''
        if self.hidden_neurons == 0:
            self.hidden_neurons_weights_in = np.random.random((1, INPUT_NODES_NUM))
            self.hidden_neurons_weights_out = np.random.random((OUTPUT_NODES_NUMBER, 1))
        else:
            # for the input hidden we need to add a row at the bottom
            self.hidden_neurons_weights_in = np.vstack([self.hidden_neurons_weights_in, np.random.random((1, INPUT_NODES_NUM))])

            # for the output layer we need to add a column
            new_matrix = np.random.random((OUTPUT_NODES_NUMBER, self.hidden_neurons + 1))
            new_matrix[:,:-1] = self.hidden_neurons_weights_out
            self.hidden_neurons_weights_out = new_matrix

        self.hidden_neurons += 1


    def calculate_move(self):
        if self.hidden_neurons_weights_in is None:
            self.output_neuron = sigmoid(np.dot(self.weights_out, self.input_neurons.T) + BIAS_VALUE)
        else:
            self.output_neuron = np.dot(self.hidden_neurons_weights_out, np.dot(self.hidden_neurons_weights_in, self.input_neurons.T))
            self.output_neuron += np.dot(self.weights_out, self.input_neurons.T)
            self.output_neuron = sigmoid(self.output_neuron)

        return calculate_direction(self.output_neuron)


    def update_input_neurons(self, position_x, position_y, distance):
        self.input_neurons = np.array([position_x, position_y, distance]).reshape(1,INPUT_NODES_NUM) / 100


