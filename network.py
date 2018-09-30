import numpy as np

'''
Values {node_index: value}
weights {(5,1): 4, (4,1): 5}
for 5 -> value += weights[edge] * values[edge[0]]
'''

# x, y distance from goal
INPUT_NODES_NUM = 5
# up down left right
OUTPUT_NODES_NUMBER = 4
BIAS_VALUE = -0.5


def sigmoid(x):
    return np.tanh(x)


def softmax(output):
    e_x = np.exp(output)
    return e_x / e_x.sum()


class NeuralNetwork:
    def __init__(self):
        self.neuron_count = INPUT_NODES_NUM + 1
        self.input_neurons = np.ones((1, INPUT_NODES_NUM), dtype=np.float32)
        self.bias = BIAS_VALUE
        self.hidden_neurons = 0
        self.weights_out = 2*np.random.random((OUTPUT_NODES_NUMBER, INPUT_NODES_NUM)) - 1
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
            self.hidden_neurons_weights_in = 2*np.random.random((1, INPUT_NODES_NUM)) - 1
            self.hidden_neurons_weights_out = 2*np.random.random((OUTPUT_NODES_NUMBER, 1)) - 1
        else:
            # for the input hidden we need to add a row at the bottom
            self.hidden_neurons_weights_in = np.vstack([self.hidden_neurons_weights_in, 2*np.random.random((1, INPUT_NODES_NUM)) - 1])

            # for the output layer we need to add a column
            new_matrix = 2*np.random.random((OUTPUT_NODES_NUMBER, self.hidden_neurons + 1)) - 1
            new_matrix[:, :-1] = self.hidden_neurons_weights_out
            self.hidden_neurons_weights_out = new_matrix

        for i in range(INPUT_NODES_NUM):
            if np.random.random(1) < 0.5:
                self.hidden_neurons_weights_in[self.hidden_neurons, i] = 0

        for i in range(OUTPUT_NODES_NUMBER):
            if np.random.random(1) < 0.5:
                self.hidden_neurons_weights_out[i, self.hidden_neurons] = 0

        self.hidden_neurons += 1


    def calculate_move(self):
        if self.hidden_neurons_weights_in is None:
            self.output_neuron = softmax(sigmoid(np.dot(self.weights_out, self.input_neurons.T) + BIAS_VALUE))
        else:
            self.output_neuron = sigmoid(np.dot(self.hidden_neurons_weights_out, np.dot(self.hidden_neurons_weights_in, self.input_neurons.T)))
            self.output_neuron += np.dot(self.weights_out, self.input_neurons.T)
            self.output_neuron = softmax(sigmoid(self.output_neuron))

        return self.output_neuron


    def update_input_neurons(self, *inputs):
        self.input_neurons = np.array(list(inputs)).reshape(1,INPUT_NODES_NUM)


    def mutate_hidden_neurons(self):
        if self.hidden_neurons == 0:
            return
        for i in range(INPUT_NODES_NUM):
            if np.random.random(1) < 0.4:
                self.hidden_neurons_weights_in[self.hidden_neurons - 1, i] = 0
            else:
                if self.hidden_neurons_weights_in[self.hidden_neurons - 1, i] == 0 and np.random.random(1) < 0.2:
                    self.hidden_neurons_weights_in[self.hidden_neurons - 1, i] = 2*np.random.random(1) - 1


        for i in range(OUTPUT_NODES_NUMBER):
            if np.random.random(1) < 0.4:
                self.hidden_neurons_weights_out[i, self.hidden_neurons - 1] = 0
            else:
                if self.hidden_neurons_weights_out[i, self.hidden_neurons - 1] == 0 and np.random.random(1) < 0.2:
                    self.hidden_neurons_weights_out[i, self.hidden_neurons - 1] = 2*np.random.random(1) - 1



