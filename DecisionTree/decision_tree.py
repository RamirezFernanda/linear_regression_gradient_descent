from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv('../DataSets/winequality-red.csv')
df['quality'] = df['quality'].replace(
    [3, 4, 5, 6, 7, 8], ['Mala', 'Mala', 'Regular', 'Regular', 'Buena', 'Buena'])

X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(random_state=23)
dt.fit(X_train, y_train)

print(f'Exactitud del modelo: {dt.score(X_test, y_test)}')
print(f'Exactitud del modelo (train): {dt.score(X_train, y_train)}')
print(
    f'Maximum distance between the root and any leaf = {DecisionTreeClassifier.get_depth(dt)}')
# print(
#    f'Parameters for this estimator. = {DecisionTreeClassifier.get_params(dt)}')
print(
    f'Number of leaves of the decision tree = {DecisionTreeClassifier.get_n_leaves(dt)}')
