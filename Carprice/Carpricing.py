import pandas as pd 

data=pd.read_csv("CarPrice.csv")
print(data.info())
print(data.describe())