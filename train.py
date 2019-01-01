import csv
import numpy as np
from sklearn.externals import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier

with open('dataset.csv', 'r') as f:
  reader = csv.reader(f)
  dataset = list(reader)

repeat_data = []
for d in dataset:
  if d[-1] == '0':
    repeat_data.append(d)


print(repeat_data)
for r in repeat_data:
  for i in range(0, 3):
    dataset.append(r)

dataset = shuffle(dataset)

dataset = np.array(dataset).astype(float)

X = dataset[:, 0:6]
y = dataset[:, 6:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,), random_state=10, max_iter=1000)
# clf = KNeighborsClassifier(3)
clf = RandomForestClassifier()

clf.fit(X_train, y_train)

y_pred = np.around(clf.predict(X_test))

print(accuracy_score(y_test, y_pred))

joblib.dump(clf, 'model')