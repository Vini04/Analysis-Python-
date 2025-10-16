import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings
import numpy as np

from sklearn.preprocessing import LabelEncoder #  encode categorical data into numerical values
from sklearn.impute import KNNImputer # impute missing values in a dataset using a k-nearest neighbors approach
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # standardize features by removing the mean and scaling to unit variance
from sklearn.metrics import f1_score # evaluate the performance of a classification model using the F1 score
from sklearn.ensemble import RandomForestRegressor #  regression model that is based upon the Random Forest model
from sklearn.ensemble import RandomForestRegressor # also to train a random forest regression model
from sklearn.model_selection import cross_val_score # used to perform k-fold cross-validation to evaluate the performance of a model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# Importing dataset
data = pd.read_csv('E:/vineet_2/python/Position_Salaries.csv')
print(data)
# Information
data.info()


# Data Preparation
# Extracting Features : extracts the features from the DataFrame and stores them in a variable named X
# Extracting Target Variable: extracts the target variable from the DataFrame and stores it in a variable named y
X = data.iloc[:,1:2].values
y = data.iloc[:,2].values


# Regressor Model
label_encoder = LabelEncoder()
x_categorical = data.select_dtypes(include=['object']).apply(label_encoder.fit_transform)
# Applies the LabelEncoder transformation to each categorical column, converting string labels into numbers
x_numerical = data.select_dtypes(exclude=['object']).values
x = pd.concat([pd.DataFrame(x_numerical), x_categorical], axis=1).values
# Combines the numerical and encoded categorical features horizontally into one dataset which is then used as input for the model.

regressor = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)
# n_estimators=10: Defines the number of decision trees in the Random Forest.
# random_state=0: Ensures the randomness in model training is controlled for reproducibility.
# oob_score=True: Enables out-of-bag scoring which evaluates the model's performance using data not seen by individual trees during training.
regressor.fit(x, y)

# Predictions and Evaluation
oob_score = regressor.oob_score #  estimates the model's generalization performance
print(f'Out-of-Bag Score: {oob_score}')

predictions = regressor.predict(x) # Makes predictions using the trained model and stores them in the 'predictions' array
# Evaluates the model's performance using the Mean Squared Error (MSE) and R-squared (R2) metrics
mse = mean_squared_error(y, predictions)
print(f'Mean Squared Error: {mse}')

r2 = r2_score(y, predictions)
print(f'R-squared: {r2}')


# Visualization
# Create a sequence of values from the minimum to maximum of the first feature (X[:, 0]), with a step size of 0.01 — this gives a smooth curve for visualization.
X_grid = np.arange(min(X[:, 0]), max(X[:, 0]), 0.01)

# Reshape X_grid to a 2D array with one column (required for model prediction)
X_grid = X_grid.reshape(-1, 1)

# Add two extra columns of zeros so that X_grid matches the original feature dimension (3 features total).
# This ensures compatibility with the trained regressor.
X_grid = np.hstack((X_grid, np.zeros((X_grid.shape[0], 2))))

# Plot the actual data points (blue dots) showing real observed values.
plt.scatter(X[:, 0], y, color='blue', label="Actual Data")

# Plot the model’s predicted values (green line) over the generated grid.
plt.plot(X_grid[:, 0], regressor.predict(X_grid), color='green', label="Random Forest Prediction")

plt.title("Random Forest Regression Results")
plt.xlabel('Position Level')
plt.ylabel('Salary')
# Add legend
plt.legend()
plt.show()



# Visualizing a Single Decision Tree from the Random Forest Model
tree_to_plot = regressor.estimators_[0] # Extract the first decision tree from the trained Random Forest model

plt.figure(figsize=(20, 10)) # Create a new figure with a custom size (width=20 inches, height=10 inches)

# Visualize the decision tree structure using matplotlib's plot_tree function
# - tree_to_plot: the decision tree to be visualized
# - feature_names: list of column names from the dataset to label the features
plot_tree(tree_to_plot, feature_names=data.columns.tolist(), filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree from Random Forest")
plt.show()