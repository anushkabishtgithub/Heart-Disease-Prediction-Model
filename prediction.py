# Importing required libraries
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Loading and reading the dataset
heart = pd.read_csv("heart_cleveland_upload.csv")

# Creating a copy of the dataset
heart_df = heart.copy()

# Renaming columns for clarity
heart_df = heart_df.rename(columns={'condition': 'target'})
print("Dataset Head:\n", heart_df.head())

# Splitting features and target
x = heart_df.drop(columns='target')
y = heart_df['target']

# Splitting dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.transform(x_test)

# Setting up the Random Forest Classifier with Grid Search for Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=5,
                           n_jobs=-1)
grid_search.fit(x_train_scaler, y_train)

# Best Model from Grid Search
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Model Evaluation
y_pred = best_model.predict(x_test_scaler)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Visualizing Feature Importance
feature_importances = best_model.feature_importances_
features = x.columns

plt.figure(figsize=(10, 6))
plt.barh(features, feature_importances, color='skyblue')
plt.xlabel("Feature Importance")
plt.title("Feature Importance in Heart Disease Prediction")
plt.show()

# Saving the Model as a Pickle File
filename = 'heart-disease-prediction-model.pkl'
pickle.dump(best_model, open(filename, 'wb'))

print("Model saved as 'heart-disease-prediction-model.pkl'.")