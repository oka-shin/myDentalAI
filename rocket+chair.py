import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sktime.classification.kernel_based import RocketClassifier
from tqdm import tqdm
import sys

np.set_printoptions(threshold=np.inf, suppress=True)

args = sys.argv
C=int(args[1])
input = np.load("./data/cat4all+chair.npy")
output8 = np.load("./data/cat4allans.npy")
result = np.zeros((input.shape[0], output8.shape[1]))
clf = RocketClassifier(num_kernels=C)

for i in tqdm(range(input.shape[0])):
    x = np.delete(input, i, 0)
    y = np.delete(output8, i, 0)
    for j in range(output8.shape[1]):
        output = y[:,j]
        clf.fit(x, output)
        yosoku = clf.predict(np.array([input[i]]))
        result[i][j] = yosoku

np.savetxt('rocket+chair-results-'+str(C)+'.txt', result, fmt='%d')
