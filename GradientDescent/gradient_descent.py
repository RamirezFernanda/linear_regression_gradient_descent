"""
El código aun no funciona aun trabajo
en ello
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def read_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    x = df[['LotFrontage', 'LotArea']].to_numpy()
    y = df['SalePrice'].to_numpy()
    return x, y


def linear_regression(x, m, b):
    return (x.dot(m)) + b


def gradient_descent(x, y, m, b, learning_rate, epochs):
    predict = linear_regression(x, m, b)
    prev_cost = 0
    error = y - predict
    for i in range(1, epochs):
        predict = linear_regression(x, m, b)
        cost = (1/x.shape[0]) * (np.sum(((y - predict)**2)))
        error = y - predict
        if not np.equal(prev_cost, cost):
            prev_cost = cost
            deriv_m = -(2/x.shape[0]) * (x.T).dot(error)
            deriv_b = -(2/x.shape[0]) * np.sum(error)
            m -= learning_rate * deriv_m
            b -= learning_rate * deriv_b

        print(f'Epoch{i}')
    return m, b


learning_rate = 0.0003
epochs = 50
x, y = read_dataset(
    'linear_regression_gradient_descent\DataSets\house_train.csv')
m = np.random.rand((x.shape[1]))
b = 0.5

m, b = gradient_descent(x, y, m, b, learning_rate, epochs)
print(f'y = {m} + {b}x')
#plt.plot(x, y, 'mo')
plt.show()
