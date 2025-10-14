full_health_data = pd.DataFrame(...)   # or load it from a CSVc
print (full_health_data.describe())

# Percentile
import numpy as np
Max_pulse = full_health_data["Max_Pulse"] # full_health_data is a DataFrame
percentile10 = np.percentile(Max_pulse, 10) # 10th percentile
print("10th Percentile of Max_Pulse:", percentile10)

# Standard Deviation
std = np.std(full_health_data)
print("Standard Deviation of Max_Pulse:", std)

# Coefficient of variation
cv = np.std(full_health_data)/np.mean(full_health_data)
print("Coefficient of Variation of Max_Pulse:", cv)

# Varience
var = np.var(full_health_data) # for full dataset
# var = np.var(full_health_data["Max_Pulse"]) # for specific column
print(var)

# Correalation Coefficient(1)
import matplotlib.pyplot as plt
health_data.plot(x ='Average_Pulse', y='Calorie_Burnage', kind='scatter')
plt.show()

# Correalation Coefficient(-1)
negative_corr = {'Hours_Work_Before_Training': [10,9,8,7,6,5,4,3,2,1], 'Calorie_Burnage': [220,240,260,280,300,320,340,360,380,400]}
negative_corr = pd.DataFrame(data=negative_corr) # convert to DataFrame

negative_corr.plot(x='Hours_Work_Before_Training', y='Calorie_Burnage', kind='scatter')
plt.show()

# No Correalation Coefficient(0)
full_health_data.plot(x='Duration', y='Max_Pulse', kind='scatter')
plt.show()

# To display correlation matrix
Corr_Matrix = round(full_health_data.corr(), 2) # round to 2 decimal places
print(Corr_Matrix)

# HeatMap
import seaborn as sns
corr_full_health_ = full_health_data.corr() # correlation matrix of full_health_data\

axis_corr= sns.heatmap(correlation_full_health, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(50, 500, n=500), square=True)
# sns.heatmap() is inbuilt function
# vmin, vmax is for color range : defines the data range that the colormap covers(max and min values)
# center is for the value at which to center the colormap when plotting divergent data
# cmap is for colormap usning seaborn.diverging_palette() function; 50 and 500 are the hue values for the start and end of the palette; n is number of colors in the palette
# square = True means that we want to see squares.
plt.show()



# Beach Example of Correlation
import pandas as pd

Drowning_Acciident = [50,70,80,120,150,180,200,250,300,350]
Ice_cream_Sales = [50,70,80,120,150,180,200,250,300,350]

Drowning = {'Drowning_Accident': Drowning_Incident, 'Ice_cream_Sales': Ice_cream_Sales}
Drowning = pd.DataFrame(data=Drowning) # convert to DataFrame

# Plot dataframe
Drowning.plot(x='Ice_cream_Sales', y='Drowning_Acciident', kind='scatter')
plt.show()

# Show correlation
corr_beach = Drowning.corr()
print(corr_beach)