import numpy as np
import matplotlib.pyplot as plt
from myfunctions import *
from utilitaire import load_data

X_train, Y_train, X_test, Y_test = load_data()
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
#show_image(X_train)

X_train, Y_train, X_test, Y_test = redimension(X_train, Y_train, X_test, Y_test)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
plt.scatter(X_train[0, :], X_train[1, :])
plt.show()

params = neural_network(X_train, Y_train, X_test, Y_test, n1=16)
fun.save_model(params)

