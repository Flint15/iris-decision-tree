# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the decision tree
decision_tree_clf = DecisionTreeClassifier(random_state=42)

param_grid = {
  'max_depth': [3, 5, 8, 10],
  'min_samples_split': [2, 3, 4],
  'max_features': [None, 'sqrt', 'log2']
}

grid_search = GridSearchCV(decision_tree_clf, param_grid, cv=5)

grid_search.fit(X_train, y_train)

best_tree = grid_search.best_estimator_

# Evaluate the model
y_pred = best_tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Visualize the tree
plt.figure(figsize=(15,10))
tree.plot_tree(best_tree, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
