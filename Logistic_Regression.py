# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Step 1 : Data Preparation
# Load the diabetes dataset
diabetes = load_diabetes()
# Defining feature matrix X and target vector y
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target)
# Convert the continuous target variable into a binary variable(1 for diabetes, 0 for no diabetes)
y = (y > y.median()).astype(int) # 1 if above median, else 0

# Step 2 : Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# test_size=0.2. : 20% of the data is used for testing, and 80% for training
# random_state=42 : ensures reproducibility of the results

# Step 3 : Feature Scaling 
# Model training and evaluation are made easier by standardization, which guarantees that the features have a mean of 0 and a standard deviation of 1
scaler = StandardScaler() # The StandardScaler instance is created; this will be used to standardize the features.
# fit_transform method : 
X_train = scaler.fit_transform(X_train) # normalize traininfr data(X_train) and determine its mean and standard deviation
X_test = scaler.transform(X_test) # standardizes the testing data (X_test) using the calculated mean and standard deviation from the training set

# Step 4: Model Training
model = LogisticRegression() # Create an instance of the LogisticRegression model
model.fit(X_train, y_train) # Fit the model to the training data
# X_train : Standardized training feature matrix
# y_train : binary target vector for the training data

# Step 5 : Model Evaluation/Prediction
y_pred = model.predict(X_test) # Predict the target variable for the test data using model.predict() method
accuracy = accuracy_score(y_test, y_pred) # Calculate the accuracy of the model by comparing the predicted values (y_pred) with the actual values (y_test)
print("Accuracy: {:.2f}%".format(accuracy * 100)) # Print the accuracy as a percentage

# Step 6 : Confusion Matrix and Classification Report
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred)) # Print the confusion matrix to evaluate the performance of the classification model
print("Classification Report:\n", classification_report(y_test, y_pred)) # Print the classification report which includes precision, recall, f1-score, and support for each class
# Confusion Matrix: A table used to describe the performance of a classification model on a set of test data for which the true values are known.
# Classification Report: A text report showing the main classification metrics

# Step 7 : Data Visualization
# Visualize the decision boundary with accuracy information
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test[:, 2], y=X_test[:, 8], hue=y_test, palette={
                0: 'blue', 1: 'red'}, marker='o')
# X_test[:, 2] : defining 'bmi' feature on x-axis 
# X_test[:, 8] : defining 's5' feature on y-axis
plt.xlabel("BMI")
plt.ylabel("Age")
plt.title("Logistic Regression Decision Boundary\nAccuracy: {:.2f}%".format(accuracy * 100))
#  {:.2f}%".format(accuracy * 100)) : formatting accuracy to 2 decimal places
plt.legend(title="Diabetes", loc="upper right")
plt.show()

# Plotting ROC Curve
#  The true positive rate (sensitivity) and false positive rate at different threshold values are determined using the probability 
#  estimates for positive outcomes (y_prob), which are obtained using the predict_proba method.
y_prob = model.predict_proba(X_test)[:, 1] # Predict the probabilities for the positive class (class 1)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
# fpr : False Positive Rate
# tpr : True Positive Rate 
# thresholds : Thresholds used to compute fpr and tpr
roc_auc = auc(fpr, tpr) # Calculate the Area Under the Curve (AUC) for the ROC curve

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
# plt.plot([0, 1], [0, 1] : plotting a diagonal line representing a random classifier
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve\nAccuracy: {:.2f}%'.format(
    accuracy * 100))
plt.legend(loc="lower right")
plt.show()