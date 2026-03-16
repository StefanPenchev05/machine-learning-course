import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./houses.csv")

# Gives the first elements from the data frame
print(data.head())

# Gives information(data types, non-null field... etc.) about the data in the data frame
print(data.info())

# Gives Important measures - mean, median, min/max .. etc.
print(data.describe())

##### Data visualization

# Histogram - reveals Distribution
# It shows how values are distributed by grouping
# numbers into ranges
plt.hist(data["Price"], bins=20)
plt.title("Price Distribution")
plt.show()

plt.scatter(data["Size"], data["Price"])
plt.xlabel("Size")
plt.ylabel("Price")
plt.show()

##### Detecting Missing Data

# Investigate missing values
print(data.isnull().sum())

# Removes missing rows
data = data.dropna()

# Replace with mean
data["Bedrooms"].fillna(data["Bedrooms"].mean(), inplace=True)