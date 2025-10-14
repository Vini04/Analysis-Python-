# Linear Regression and Regression table with one explanatory variable

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf

# Step 1: Data Preparation : data cleaning, filtering, handling NaN`s
# importing the dataset
full_health-data = pd.read_csv('full_health_data.csv', header=0, sep=',')

# defining the variables
x = full_health_data['Average_Pulse']
y = full_health_data['calorie_Burnage']

# Handling NaN`s and filtering values
full_health_data.dropna(axis=0, inplace=True)  # Remove rows with missing values
full_health_data = full_health_data[full_health_data['Average_Pulse'] > 0]  # Filter rows where 'Average_Pulse' > 0
full_health_data = full_health_data[full_health_data['Calorie_Burnage'] > 0]  # Filter rows where 'Calorie_Burnage' > 0

# Step 2: Data Visualization
# Scatter plot to visualize the relationship between Average_Pulse and Calorie_Burnage
# performing linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y) 
# r_value is the correlation coefficient
# p_value is the two-sided p-value for a hypothesis test whose null hypothesis is that the slope is zero
# std_err is the standard error of the estimated gradient; standard error gives an idea of how precise the estimate is
# stats.linregress() returns a tuple containing these values

# define the regression line function
def myfunc(x):
    return slope * x + intercept

# create the regression line values
mymodel = list(map(myfunc, x))
# list(map()) applies the function to each item in the iterable (x) and returns a list of the results

# Step 3: Data Analysis
# plotting the scatter plot and the regression line
plt.scatter(x, y) # scatter plot
plt.plot(x, mymodel, color='red') # regression line
plt.ylim(ymin(0), y.max(2000))  # setting the y-axis limits
plt.xlim(x.min(0), x.max(200))  # setting the x-axis limits
plt.xlabel('Average Pulse')
plt.ylabel('Calorie Burnage')
plt.show()

# Regression Table
model = smf.ols(formula='Calorie_Burnage ~ Average_Pulse', data=full_health_data) # defining the model; explanatory variable must be written first in the parenthesis
results = model.fit() # fitting the model to obtain the variable results
print(results.summary()) # printing the regression table 

# Define the linear regression function in Python to perform predictions.
# What is Calorie_Burnage if Average_Pulse is: 120, 130, 150, 180?
def Predict_Calorie_Burnage(Average_Pulse):
    # return slope * Average_Pulse + intercept
    return (0.3296 * Average_Pulse) + 346.8662 # using the coefficients from the regression table

print("Predicted Calorie Burnage for Average Pulse of 120:", Predict_Calorie_Burnage(120))
print("Predicted Calorie Burnage for Average Pulse of 130:", Predict_Calorie_Burnage(130))
print("Predicted Calorie Burnage for Average Pulse of 150:", Predict_Calorie_Burnage(150))
print("Predicted Calorie Burnage for Average Pulse of 180:", Predict_Calorie_Burnage(180))