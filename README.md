# Getting started

## Prerequisites
* numpy
* pandas
* matplotlib
* scikit-learn
  
## Instalation
`pip install numpy`

`pip install pandas`

`pip install matplotlib`

`pip install scikit-learn`

## Usage

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
```
`from sklearn import tree` use for visualization of decision trees
### Load, split and train
```python
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

decision_tree_clf = DecisionTreeClassifier(random_state=42)
decision_tree_clf.fit(X_train, y_train)
```
`decision_tree_clf = DecisionTreeClassifier(random_state=42)` Make the results of the decision tree classifier are reproducible
### Visualisation of prediction
```python
y_pred = decision_tree_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

plt.figure(figsize=(15,10))
tree.plot_tree(decision_tree_clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
```
`plt.figure(figsize=(15,10))` is used to specify the size of the figure

`tree.plot_tree(decision_tree_clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
` used to visualize the decision tree

`filled=True` Fills the nodes in the plot with colors to indicate the majority class in each node
