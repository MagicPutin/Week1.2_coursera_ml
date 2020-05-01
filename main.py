import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Обратите внимание, что признак Sex имеет строковые значения.
row_data = pd.read_csv('data/titanic.csv', index_col='PassengerId')

# delete nan age
row_data = row_data.dropna(subset=['Age'])

#################################################
# NEED to transfer female and male into 1 and 0 #
#################################################

data = row_data[['Pclass', 'Fare', 'Age', 'Sex']]
goal = row_data['Survived']

clf = DecisionTreeClassifier()
clf.fit(data, goal)
print(clf.feature_importances_)