import torch
import glob
import unicodedata
import string
import numpy as np

def findFiles(path): return glob.glob(path)

# Read a file and split into lines
def read_data(filename):
    data = np.load(filename)
    print(data.shape)
    return data

# Build the category_lines dictionary, a list of lines per category
category_lines = []
all_categories = []
for filename in findFiles('../training/motion/*.npy'):
    category = filename.split('_')[-2]
    all_categories.append(category)
    data = read_data(filename)
    category_lines.append(data) #此时数据data为n*900的nparray
    

n_categories = len(all_categories)

flag_lines = []
for i in range(n_categories):
    flag = np.ones(category_lines[i].shape[0]) * i
    flag_lines.append(flag)

feature_set = np.concatenate(category_lines)
flag_set = np.concatenate(flag_lines)
print(feature_set.shape, flag_set.shape)