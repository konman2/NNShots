import numpy as np
x = np.load("./data/X.npy")
y = np.load("./data/y.npy").T

print(np.mean(y))
