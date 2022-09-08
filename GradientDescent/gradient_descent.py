"""
El código aun no funciona aun trabajo
en ello
"""
from sklearn.preprocessing import MinMaxScaler
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def read_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df.to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=[
        'No', 'X1 transaction date',
        'X2 house age', 'X3 distance to the nearest MRT station',
        'X4 number of convenience stores',
        'X5 latitude', 'X6 longitude', 'Y house price of unit area'])
    x = df_scaled['X2 house age']
    y = df_scaled['Y house price of unit area']
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
        #print(f'Epoch{i} Error{error} b{b} m{m}')
    return m, b


learning_rate = 0.01
epochs = 1000
x, y = read_dataset(
    'DataSets/real_estate.csv')
m = np.random.rand(x.shape[0])
b = 0.5

m, b = gradient_descent(x, y, m, b, learning_rate, epochs)
y_final = m + b * x
print(f'y = {m} + {b}x')
print(x)
plt.scatter(x, y)
plt.plot([min(x), max(x)], [min(y_final), max(y_final)])
plt.show()
