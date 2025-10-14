import numpy as np

Max_value = max(1,2,3,4,5,6,7,8,9,10)
print("Maximum Value:", Max_value)

Min_value = min(1,2,3,4,5,6,7,8,9,10)
print("Minimum Value:", Min_value)

Calorie_burnage = [100, 200, 300, 400, 500]
Avg_calorie_burnage = np.mean(Calorie_burnage)
print("Average Calorie Burnage:", Avg_calorie_burnage) #  np. in front of mean to let Python know that we want to activate the mean function from the Numpy library
