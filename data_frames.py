import pandas as pd
d = {'col1': [1, 2, 3, 4], 'col2': [5, 6, 7, 8], 'col3': [9, 10, 11, 12]}
df = pd.DataFrame(data=d)
print("DataFrame:\n", df) #  pd. in front of DataFrame() to let Python know that we want to activate the DataFrame() function from the Pandas library

count_column = df.shape[1] # Count the number of columns in the DataFrame
print ("Count of columns in DataFrame:", count_column)

count_row = df.shape[0] # Count the number of rows in the DataFrame
print ("Count of rows in DataFrame:", count_column)