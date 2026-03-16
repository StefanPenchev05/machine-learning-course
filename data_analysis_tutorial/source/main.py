import pandas as pd

data = pd.read_csv("./houses.csv")

# Gives the first elements from the data frame
print(data.head())

# Gives information(data types, non-null field... etc.) about the data in the data frame
print(data.info())
