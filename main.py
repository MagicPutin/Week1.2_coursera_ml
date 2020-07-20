
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# Обратите внимание, что признак Sex имеет строковые значения.
row_data = pd.read_csv('data/titanic.csv', index_col='PassengerId')

# delete nan age
row_data = row_data.dropna(subset=['Age'])

# transfer sex to 0 or 1
row_data['Sex'] = row_data['Sex'].transform(func=lambda x: 0 if (x == 'female') else 1)
# guess it's work

data = row_data[['Pclass', 'Fare', 'Age', 'Sex']]
goal = row_data['Survived']

clf = DecisionTreeClassifier()
clf.fit(data, goal)
print(clf.feature_importances_)
task = open('answers/task.txt', 'w')
task.write('Fare Sex')
task.close()