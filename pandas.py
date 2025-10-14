import pandas as pd

# Load a CSV file into a DataFrame
health_data = pd.read_csv("data.csv", header=0, sep=',')
# header 0 means that the first row of the CSV file contains the column names
# sep ',' means that the values in the CSV file are separated by commas
print("DataFrame:\n", health_data)

# Describe
print("DataFrame Description:\n", health_data.describe())

# Group by
grouped_data = health_data.groupby('Calories').mean()  # Group by 'Calories' column and calculate the mean of other columns
print("Grouped Data by Calories (mean):\n", grouped_data)

# Merge
df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
df2 = pd.DataFrame({'A': [1, 2, 3], 'C': [7, 8, 9]})
merged_df = pd.merge(df1, df2, on='A')  # Merge df1 and df2 on column 'A'

# Filtering and Handling NaN`s
filtered_data = health_data[health_data['Calories'] > 500]  # Filter rows where 'Calories' > 500
print("Filtered Data (Calories > 500):\n", filtered_data) # NaN`s are handled automatically by pandas using : fillna(), dropna(), isna()
print("Merged DataFrame:\n", merged_df)

# Remove rows with missing values
health_data.dropna(axis=0, inplace=True)  # Remove rows with missing values
# inplace=True means that the changes will be applied to the original DataFrame
print("DataFrame after dropping missing values:\n", health_data)

# to show datatypes within our dataset
print(health_data.info())

# to convert into float64
health_data["Average_Pulse"] = health_data["Average_Pulse"].astype('float64')
health_data["Max_Pulse"] = health_data["Max_Pulse"].astype('float64')
print(health_data.info())

# to analyse the data
print("Descriptive statistics:\n", health_data.describe())



