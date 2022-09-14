from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from mlxtend.evaluate import bias_variance_decomp


df = pd.read_csv('../DataSets/real_estate.csv')

X = df.drop(columns=['Y house price of unit area',
            'No'], axis=1)
y = df['Y house price of unit area']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

lr = make_pipeline(StandardScaler(), LinearRegression())
lr.fit(X_train, y_train)

predictions = lr.predict(X_test)

print('Parametros utiles para el analisis del modelo')
print('/------------------------------/')
print(f'Exactitud del modelo: {lr.score(X_test, y_test)}')
print('/------------------------------/')
print(f'Exactitud del modelo (train): {lr.score(X_train, y_train)}')
print('/------------------------------/')
print(
    f'm (coeficientes)= {lr.steps[1][1].coef_}')
print('/------------------------------/')
print(
    f'b = {lr.steps[1][1].intercept_}')
print('/------------------------------/')
print("Error medio cuadrado: %.2f" %
      mean_squared_error(y_test, predictions))
print('/------------------------------/')
print("Coeficiente de determinación (score): %.2f" %
      r2_score(y_test, predictions))
print('/------------------------------/')
print('/------------------------------/')
print('/------------------------------/')
pruebas = pd.DataFrame(
    {'Valor esperado': y_test, 'Valor arrojado': predictions})
print(f'Predicciones: {pruebas}')
print('/------------------------------/')
print('/------------------------------/')
print('/------------------------------/')
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
mse, bias, var = bias_variance_decomp(
    lr, X_train, y_train, X_test, y_test, loss='mse', num_rounds=200, random_seed=123)
print('MSE de bias_variance lib [pérdida promedio esperada]: %.3f' % mse)
print('/------------------------------/')
print('Sesgo promedio: %.3f' % bias)
print('/------------------------------/')
print('Varianza promedio: %.3f' % var)
print('/------------------------------/')
print('/------------------------------/')
print('/------------------------------/')
