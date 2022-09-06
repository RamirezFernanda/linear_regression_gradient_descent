"""
El código aun no funciona por lo que 
pedire una asesoria :c
"""

import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def read_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    x = df['Weight']
    y = df['Length1']
    return x, y


def error(x, y, a, b):
    m = len(x)
    error = 0.0
    for i in range(m):
        hypothesis = a+b*x[i]
        error += (y[i] - hypothesis) ** 2
    return error / (2*m)


def gradient_descent(x, y, a, b, learning_rate, epochs):
    pass


a = 1
b = 1
learning_rate = 0.001
epochs = 10
x, y = read_dataset(
    'linear_regression_gradient_descent\DataSets\Fish.csv')

a, b = gradient_descent(x, y, a, b, learning_rate, epochs)
print(f'y = {a} + {b}x')
plt.plot(x, y, 'mo')
plt.show()
