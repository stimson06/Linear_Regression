
from word2number import w2n
from sklearn import linear_model
import pandas as pd
import numpy as np

df=pd.read_csv("hiring.csv")
df.fillna(method='bfill',inplace=True)
df.experience = df['experience'].apply(w2n.word_to_num)
print (df)

reg=linear_model.LinearRegression()
reg.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df.salary)
print("The salary of the candidate :",int(reg.predict([[2,9,6]])))
print ("The salary of the candidate :",int(reg.predict([[15,10,10]])))