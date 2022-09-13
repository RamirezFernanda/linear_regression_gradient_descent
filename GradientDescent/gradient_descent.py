from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def read_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=[
        'No', 'X1 transaction date',
        'X2 house age', 'X3 distance to the nearest MRT station',
        'X4 number of convenience stores',
        'X5 latitude', 'X6 longitude', 'Y house price of unit area'])
    df_train = df_scaled.sample(frac=0.8, random_state=25)
    df_test = df_scaled.drop(df_train.index)

    X_train = df_train[['X5 latitude', 'X6 longitude']]
    y_train = df_train['Y house price of unit area']

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    y_train = np.reshape(y_train, (y_train.shape[0],))

    X_test = df_test[['X5 latitude', 'X6 longitude']]
    y_test = df_test['Y house price of unit area']

    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    return X_train, y_train, X_test, y_test


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


# Read dataset
X_train, y_train, X_test, y_test = read_dataset(
    '../DataSets/real_estate.csv')
# Initialize learning_rate, epochs, m, and b
learning_rate = 0.01
epochs = 7000
m, b = np.random.rand((X_train.shape[1])) * 10, np.random.random()
# Train model
m, b = gradient_descent(X_train, y_train, m, b, learning_rate, epochs)
# Make predictions
pred = linear_regression(X_test, m, b)
# Compare predictions vs expected value
results = pd.DataFrame({'Valor esperado': y_test, 'Valor dado': pred})
print('/--------------------------------------------------/')
print('Predicciones')
print(results)
print('/--------------------------------------------------/')


def determination_coefficient(x, y):
    a = np.sum((x - np.mean(x)) ** 2)
    b = np.sum((y - np.mean(x)) ** 2)
    return 1 - (b / a)


print(f'Exactitud del modelo: {determination_coefficient(y_test, pred)}')
