import numpy as np
from sklearn.svm import SVC

f = open('svm_data.txt', 'r')
lines = f.readlines()
f.close()
X = []
y = []
for line in lines:
    conts = line.split()
    motion_type = int(conts[0])
    data = []
    contslen = len(conts)
    for i in range(1, contslen):
        data.append(float(conts[i]))
    y.append(motion_type)
    X.append(data)

# X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
# y = np.array([1, 1, 2, 3])
clf = SVC(gamma='auto')
clf.fit(X, y) 
