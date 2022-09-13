from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


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
