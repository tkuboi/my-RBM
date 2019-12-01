"""utils.py
Contains utility functions
"""

import numpy as np

class Dataset:
    def __init__(self, inputs=None, targets=None):
        self.inputs = inputs
        self.targets = targets


def load_data(filename):
    inputs = []
    with open(filename) as fo:
        for line in fo:
            items = line.split()
            inputs.extend([to_vector(item.replace('\n', '')) for item in items])
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

