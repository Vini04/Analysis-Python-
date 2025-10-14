# Linear Regression and Regression table with multiple explanatory variable

import pandas as pd
import statsmodels.formula.api as smf

# importing the dataset
full_health_data = pd.read_csv('full_health_data.csv', header=0, sep=',')

# Regression Table plotting
model = smf.ols(formula='Calorie_Burnage ~ Average_Pulse + Duration', data=full_health_data) # defining the model; explanatory variable must be written first in the parenthesis
results = model.fit() # fitting the model to obtain the variable results
print(results.summary()) # printing the regression table


# Define the linear regression function in Python to perform predictions.
def Predict_Calorie_Burnage(Average_Pulse, Duration):
 return(3.1695*Average_Pulse + 5.8434 * Duration - 334.5194)

print(Predict_Calorie_Burnage(110,60))
print(Predict_Calorie_Burnage(140,45))
print(Predict_Calorie_Burnage(175,20))