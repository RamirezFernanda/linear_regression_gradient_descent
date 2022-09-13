from sklearn.preprocessing import MinMaxScaler
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
    x = df_scaled[['X5 latitude', 'X6 longitude']]
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
        else:
            break
    return m, b


def train_test(x, y, test_percentage):
    X_test = []
    y_test = []
    X_train = []
    y_train = []
    test_size = round(len(x) * test_percentage)

    for i in range(len(x)):
        if len(X_test) < test_size:
            X_test.append(x[i])
            y_test.append(y[i])
        else:
            X_train.append(x[i])
            y_train.append(y[i])

    return X_test, X_train, y_test, y_train


X_test, X_train, y_test, y_train = train_test(x, y, 0.2)


learning_rate = 0.002
epochs = 7000
x, y = read_dataset(
    '../DataSets/real_estate.csv')
print(x.shape)
print(y.shape)
m = np.zeros(x.shape[1])
b = 0

m, b = gradient_descent(x, y, m, b, learning_rate, epochs)
y_final = m + b * x
print(f'y = {m} + {b}x')
"""plt.scatter(x[1], y)
plt.plot([min(x), max(x)], [min(y_final), max(y_final)])
plt.show()"""
