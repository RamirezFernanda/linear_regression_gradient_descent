from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


df = pd.read_csv('../DataSets/real_estate.csv')

X = df.drop(columns=['Y house price of unit area', 'No'], axis=1)
y = df['Y house price of unit area']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

lr = make_pipeline(StandardScaler(), LinearRegression())
lr.fit(X_train, y_train)

print(f'Exactitud del modelo: {lr.score(X_test, y_test)}')
print(f'Exactitud del modelo (train): {lr.score(X_train, y_train)}')
print(
    f'm = {lr.steps[1][1].coef_}')
# print(
#    f'Parameters for this estimator. = {DecisionTreeClassifier.get_params(lr)}')
print(
    f'b = {lr.steps[1][1].intercept_}')
