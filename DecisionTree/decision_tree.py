from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv('../DataSets/winequality-red.csv')
df['quality'] = df['quality'].replace(
    [3, 4, 5, 6, 7, 8], ['Mala', 'Mala', 'Regular', 'Regular', 'Buena', 'Buena'])

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(random_state=23)
dt.fit(X_train, y_train)

print(f'Exactitud del modelo: {dt.score(X_test, y_test)}')
print('/------------------------------/')
print(f'Exactitud del modelo (train): {dt.score(X_train, y_train)}')
print('/------------------------------/')
print(
    f'Distancia máxima entre la raíz y cualquier hoja = {DecisionTreeClassifier.get_depth(dt)}')
print('/------------------------------/')
# print(
#    f'Parameters for this estimator. = {DecisionTreeClassifier.get_params(dt)}')
print(
    f'Número de hojas del arbol = {DecisionTreeClassifier.get_n_leaves(dt)}')
print('/------------------------------/')
print('/------------------------------/')
print('/------------------------------/')

predictions = dt.predict(X_test)
pruebas = pd.DataFrame(
    {'Valor esperado': y_test, 'Valor obtenido': predictions})
print(f'Predicciones: {pruebas}')
