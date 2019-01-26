import torch
import glob
import unicodedata
import string
import numpy as np

all_letters = string.ascii_letters + " .,;'-"
n_letters = 18#每次输入18维的数据，长度固定为50

def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def read_data(filename):
    data = np.load(filename)
    return [data[i] for i in range(data.shape[0])]

# Build the category_lines dictionary, a list of lines per category
category_lines = {}
all_categories = []
for filename in findFiles('../training/motion/*.npy'):
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    data = read_data(filename)
    category_lines[category] = data #此时数据data为n*900的list

n_categories = len(all_categories)

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    data_length = 50
    tensor = torch.zeros(data_length, 1, n_letters)
    data_reform = np.array(line).reshape(50, 18)
    for i in range(data_length):
        for j in range(n_letters):
            tensor[i][0][j] = data_reform[i, j]
    return tensor
    # for li, letter in enumerate(line):
    #     tensor[li][0][letterToIndex(letter)] = 1
    # return tensor

