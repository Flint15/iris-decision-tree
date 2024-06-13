# Grid search

## Changes

```python
# import necessary library
from sklearn.model_selection import GridSearchCV

#Define the parameter grid for GridSearchCV
param_grid = {
  'max_depth': [3, 5, 8, 10],
  'min_samples_split': [2, 3, 4],
  'max_features': [None, 'sqrt', 'log2']
}

# Initialize GridSearchCv
grid_search = GridSearchCV(decision_tree_clf, param_grid, cv=5) # cv-cross validation

# Fit the model using GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_tree = grid_search.best_estimator_

# Predict using the best model
y_pred = best_tree.predict(X_test)

print(f"Best Parameters: {grid_search.best_params_}")

# Visualize the best decision tree
tree.plot_tree(best_tree, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
```

