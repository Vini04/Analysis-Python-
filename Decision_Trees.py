# Building Phase
#  Preprocess the dataset.
#  Split the dataset from train and test using Python sklearn package.
#  Train the classifier.

# Operational Phase
#  Make predictions.
#  Calculate the accuracy.

# Dataset :
# Title  : Balance Scale Weight & Distance 
#Database
# Number of Instances  : 625 (49 balanced, 288 left, 288 right)
# Number of Attributes  : 4 (numeric) + class name = 5
#Attribute Information:
# 1. Class Name (Target variable): 3
       # L [balance scale tip to the left]
       # B [balance scale be balanced]
       # R [balance scale tip to the right]
# 2. Left-Weight: 5 (1, 2, 3, 4, 5)
# 3. Left-Distance: 5 (1, 2, 3, 4, 5)
# 4. Right-Weight: 5 (1, 2, 3, 4, 5)
# 5. Right-Distance: 5 (1, 2, 3, 4, 5)
# Missing Attribute Values: None
# Class Distribution:
      # 1. 46.08 percent are L
      # 2. 07.84 percent are B
      # 3. 46.08 percent are R

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import tree
import seaborn as sns
import numpy as np


# Data Import and Exploration
url = "https://archive.ics.uci.edu/machine-learning"
# Defining the column names
column_names = ['Class', 'Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance'] # Heaader names for the dataset
data = pd.read_csv('databases/balance-scale/balance-scale.data',url, sep='', header=None) # Reading the dataset into a pandas DataFrame

print("Dataset Length: ", len(data)) # Prints the number of rows in the dataset
print("Dataset Shape: ", data.shape)
print("Dataset: ", data.head())


# Train Test Split
# Separating the target variable
X = data.values[:, 1:5] # Features are all columns except the first one (Class)
Y = data.values[:, 0] # Target variable is the first column (Class)
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
# test_size=0.3 means 30% of the data will be used for testing
# Training using gini index
clf_gini = DecisionTreeClassifier(criterion="gini",random_state=100, max_depth=3, min_samples_leaf=5)
# decision tree classifier object is created with gini index as the criterion for splitting, a random state for reproducibility, a maximum depth of 3, and a minimum of 5 samples required to be at a leaf node
clf_gini.fit(X_train, y_train) # # Performing training
# Training using entropy
clf_entropy = DecisionTreeClassifier(criterion="entropy",random_state=100, max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train) # Performing training


# Predcition and Evaluation
# Function
def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test) # Predicting the class labels for the test set
    print("Predicted values:") 
    print(y_pred) # Printing the predicted class labels
    return y_pred
# This function defines the prediction() function, which is responsible for making predictions on the test data using the trained classifier object. 
# It passes the test data to the classifier's predict() method and prints the predicted class labels.

# Placeholder function for cal_accuracy
def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ",
          confusion_matrix(y_test, y_pred))
    print("Accuracy : ",
          accuracy_score(y_test, y_pred)*100)
    print("Report : ",
          classification_report(y_test, y_pred))
# This function defines the cal_accuracy() function, which is responsible for calculating the accuracy of the predictions. 
# It calculates and prints the confusion matrix, accuracy score, and classification report, providing detailed performance evaluation.


# Plotting the decision tree
def plot_decision_tree(clf_object, feature_names, class_names):
    plt.figure(figsize=(15, 10))
    plot_tree(clf_object, filled=True, feature_names=feature_names, class_names=class_names, rounded=True)
    plt.show()
# clf_object: The trained decision tree model object.
# filled=True: This argument fills the nodes of the tree with different colors based on the predicted class majority.
# feature_names: This argument provides the names of the features used in the decision tree.
# class_names: This argument provides the names of the different classes.
# rounded=True: This argument rounds the corners of the nodes for a more aesthetically pleasing appearance.

if __name__ == "__main__":
    data()
    X, Y, X_train, X_test, y_train, y_test = data

    clf_gini = X_train, X_test, y_train
    clf_entropy = X_train, X_test, y_train

    # Visualizing the Decision Trees
    plot_decision_tree(clf_gini, ['X1', 'X2', 'X3', 'X4'], ['L', 'B', 'R'])
    plot_decision_tree(clf_entropy, ['X1', 'X2', 'X3', 'X4'], ['L', 'B', 'R'])
# if __name__ == "__main__": ensures that the code block runs only when the script is executed directly, not when imported as a module.


# Results using Gini Index
print("Results Using Gini Index:")
y_pred_gini = prediction(X_test, clf_gini) # Making predictions on the test set using the gini index classifier
cal_accuracy(y_test, y_pred_gini) # Calculating and printing the accuracy of the gini index classifier

# Results using Entropy
print("Results Using Entropy:")
y_pred_entropy = prediction(X_test, clf_entropy) # Making predictions on the test set using
cal_accuracy(y_test, y_pred_entropy) # Calculating and printing the accuracy of the entropy classifier the entropy classifier