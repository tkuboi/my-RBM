"""
"""
import numpy as np
import math
import random
import sys

class Dataset:
    def __init__(self, inputs=None, targets=None):
        self.inputs = inputs
        self.targets = targets

class RBM:
    def __init__(self, n_hid, report=False):
        self.n_hid = n_hid
        self.report_calls_to_sample_bernoulli = report 
        self.randomness_source = self.generate_randomness_source(self.n_hid * 10000) 

    def generate_randomness_source(self, size):
        arr = [random.random() for i in range(size)] 
        return np.array(arr)

    def cd1(self, rbm_w, visible_data):
        """ <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
        <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
        The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
        """
        visible_data = self.sample_bernoulli(visible_data)
        hidden_state = self.visible_state_to_hidden_probabilities(rbm_w, visible_data)
        #goodness1 = self.configuration_goodness(rbm_w, visible_data, hidden_state)
        hidden_state = self.sample_bernoulli(hidden_state)
        grad1 = self.configuration_goodness_gradient(visible_data, hidden_state)
        rbm_w = rbm_w + grad1
    
        visible_state = self.hidden_state_to_visible_probabilities(rbm_w, hidden_state)
        visible_state = self.sample_bernoulli(visible_state)
        hidden_state = self.visible_state_to_hidden_probabilities(rbm_w, visible_state)
        #hidden_state = sample_bernoulli(hidden_state)
        grad2 = self.configuration_goodness_gradient(visible_state, hidden_state)
        #goodness2 = configuration_goodness(rbm_w, visible_state, hidden_state)
        #gap = goodness1 - goodness2
        #grad = configuration_goodness_gradient(visible_state, hidden_state)
        return grad1 - grad2

    def optimize(self, gradient_function, training_data, learning_rate, n_iterations):
        # This trains a model that's defined by a single matrix of weights.
        # <model_shape> is the shape of the array of weights.
        # <gradient_function> is a function that takes parameters <model> and <data> and returns the gradient (or approximate gradient in the case of CD-1) of the function that we're maximizing. Note the contrast with the loss function that we saw in PA3, which we were minimizing. The returned gradient is an array of the same shape as the provided <model> parameter.
        # This uses mini-batches of size 100, momentum of 0.9, no weight decay, and no early stopping.
        # This returns the matrix of weights of the trained model.
        model_shape = np.array([self.n_hid, 900])
        model = (self.rand(model_shape, model_shape.prod()) * 2 - 1) * 0.1
        momentum_speed = np.zeros(model_shape)
        mini_batch_size = 100
        start_of_next_mini_batch = 0 
        for iteration_number in range(1, n_iterations):
            mini_batch = self.extract_mini_batch(training_data, start_of_next_mini_batch, mini_batch_size)
            start_of_next_mini_batch = start_of_next_mini_batch + mini_batch_size % training_data.inputs.shape[1]
            gradient = gradient_function(model, mini_batch.inputs)
            momentum_speed = 0.9 * momentum_speed + gradient
            model = model + momentum_speed * learning_rate
        return model

    def visible_state_to_hidden_probabilities(self, rbm_w, visible_state):
        """ <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
        <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
        The returned value is a matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
        This takes in the (binary) states of the visible units, and returns the activation probabilities of the hidden units conditional on those states.
        """
        #print(rbm_w.shape)
        #print(visible_state.shape)
        #print(visible_state)
        return 1 / (1 + np.exp(np.matmul(-rbm_w, visible_state))) 

    def hidden_state_to_visible_probabilities(self, rbm_w, hidden_state):
        """<rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
        <hidden_state> is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
        The returned value is a matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
        This takes in the (binary) states of the hidden units, and returns the activation probabilities of the visible units, conditional on those states.
        """
        return 1 / (1 + np.exp(np.matmul(-rbm_w.T, hidden_state)))

    def sample_bernoulli(self, probabilities):
        if self.report_calls_to_sample_bernoulli:
            print('sample_bernoulli() was called with a matrix of size %d by %d. ', probabilities.shape[1], probabilities.shape[2])
        seed = np.sum(probabilities)
        #print("seed=", seed)
        binary = (probabilities > self.rand(np.array(list(probabilities.shape)), seed)) # the "+" is to avoid the "logical" data type, which just confuses things.
        return binary.astype(int)

    def configuration_goodness(rbm_w, visible_state, hidden_state):
        """<rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
         <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
         <hidden_state> is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
         This returns a scalar: the mean over cases of the goodness (negative energy) of the described configurations.
        """
        return sum(sum(rbm_w * visible_state * hidden_state)) / columns(visible_state)

    def configuration_goodness_gradient(self, visible_state, hidden_state):
        return (hidden_state * visible_state.T) / visible_state.shape[1] 

    """def show_rbm(self, rbm_w):
        n_hid = rbm_w.shape(1)
        n_rows = math.ceil(sqrt(n_hid))
        blank_lines = 4
        distance = 16 + blank_lines
        to_show = np.zeros([n_rows * distance + blank_lines, n_rows * distance + blank_lines])
        for i in range(n_hid-1):
            row_i = math.floor(i / n_rows)
            col_i = i % n_rows
            pixels = np.reshape(rbm_w(i+1, :), [16, 16]).T
            row_base = row_i*distance + blank_lines
            col_base = col_i*distance + blank_lines
            to_show(row_base+1:row_base+16, col_base+1:col_base+16) = pixels
        extreme = max(abs(to_show(:)))
        try:
            imshow(to_show, [-extreme, extreme])
            title('hidden units of the RBM')
        except: 
            printf('Failed to display the RBM. No big deal (you do not need the display to finish the assignment), but you are missing out on an interesting picture.\n')
    """

    def rand(self, requested_size, seed):
        start_i = int(round(seed) % round(self.randomness_source.shape[0] / 10) + 1)
        if start_i + requested_size.prod() >= self.randomness_source.shape[0] + 1:
            raise ValueError('a4_rand failed to generate an array of that size (too big)')
        #print(start_i, requested_size.prod())
        return np.reshape(self.randomness_source[start_i : start_i + requested_size.prod()], requested_size)

    def extract_mini_batch(self, data_set, start_i, n_cases):
        mini_batch = Dataset()
        mini_batch.inputs = data_set.inputs[:, start_i : start_i + n_cases]
        mini_batch.targets = data_set.targets[:, start_i : start_i + n_cases]
        return mini_batch

def load_data(filename):
    inputs = []
    with open(filename) as fo:
        for line in fo:
            items = line.split('\t')
            inputs.extend([to_vector(item) for item in items])
    return Dataset(np.array(inputs).T, np.array(inputs).T) 

def to_vector(word):
    vec = np.zeros((30, 30)) 
    for i, char in enumerate(word.lower()):
        if char.isalpha():
            vec[i][ord(char) - ord('a')] = 1
        elif char == "'":
            vec[i][26] = 1
        elif char == "-":
            vec[i][27] = 1
    return vec.reshape(900)

def main():
    filename = sys.argv[1]
    n_hid = 300
    rbm = RBM(n_hid)
    lr_rbm = .02
    n_iterations = 10
    dataset = load_data(filename)
    #print(dataset.inputs.shape)
    #print(dataset.inputs)
    #print(dataset.targets)
    model = rbm.optimize(rbm.cd1, dataset, lr_rbm, n_iterations)
    print(model)

if __name__ == '__main__':
    main()

