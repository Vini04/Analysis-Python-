import matplotlib.pyplot as plt

# Plot a line graph
health_data.plot(x-'Average_Pulse', y= 'Calorie_Burnage', kind='line'),
# Set x and y axis limits
plt.ylim(ylim = 0) # Set y axis limit to start from 0
plt.xlim(xlim = 0) # Set x axis limit to start from 0
# To format axis
# plt.ylim(ymin=0, ymax=400)
# plt.xlim(xmin=0, xmax=150)

plt.show() 

#  Slope of a line
def slope(x1, y1, x2, y2):
  s = (y2-y1)/(x2-x1)
  return s

print (slope(80,240,90,260))

# Slope and intercept of a line using python function
import numpy as np
import pandas as pd

health_data = pd.read_csv("data.csv", header=0, sep=',')

x = health_data["Average_Pulse"]
y = health_data["Calorie_Burnage"]
slope_intercept = np.polyfit(x, y, 1) # 1 means linear(degree 1)
print(slope_intercept) # slope and intercept

# Slope and intercept of a line using python(define) function
def slope_intercept(x):
  return 2*x + 80
print(slope_intercept(90))