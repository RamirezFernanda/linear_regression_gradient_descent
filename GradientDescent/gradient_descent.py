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
    n = len(x)
    for i in range(epochs):
        y_predicted = a * x + b
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        a = a - learning_rate * md
        b = b - learning_rate * bd
        print(f'epoch{i}')
    return a, b


a = 1
b = 1
learning_rate = 0.001
epochs = 10000
x, y = read_dataset(
    'linear_regression_gradient_descent\DataSets\Fish.csv')

a, b = gradient_descent(x, y, a, b, learning_rate, epochs)
print(f'y = {a} + {b}x')
plt.plot(x, y, 'mo')
plt.show()
