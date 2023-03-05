import numpy as np
import matplotlib.pyplot as plt
from myfunctions import *
from utilitaire import load_data
from sklearn.datasets import make_circles


X, y = make_circles(n_samples=100, random_state=0,  noise = 0.01)

y = y.reshape(y.shape[0], 1)

X = X.T
Y = y.T

print('X', X.shape)
print('Y', Y.shape)

params, score = neural_network(X, Y, hidden_layers = (32, 32,  32))

if score > 98:
    save_model(params)
    print("successful backup of model")