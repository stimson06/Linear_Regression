import pandas as pd 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_data(path):
    data=pd.read_csv(path)
    #print(data.info())
    missing_val=(data.isna().sum()) #Counts of missing values before imputation
    #print(missing_val)
    if (sum(missing_val)>0):
        reframeing=Imputation(data,missing_val)
    #missing_val=(data.isna().sum()) # Counts of missing values after imputation 
    #print(missing_val)
    return data
    
def Imputation(data,missing_val):
    print("Impuation of numeric data ....")
    col_numeric = list(data.select_dtypes(exclude="object")) # Columns with numerical datatype
    imputed_data=pd.DataFrame(data,columns=col_numeric)
    Impute=IterativeImputer(max_iter=10, random_state=610)
    Data=pd.DataFrame(Impute.fit_transform(pd.DataFrame(imputed_data)),columns=col_numeric)
    print("Imputation completed....")
    data[col_numeric]=Data[col_numeric]
    return data

def Visualization(data):
    col_categorical = list(data.select_dtypes(include="object")) # Columns with numerical datatype
    
    #Visualization of categorical datatypes
    plt.figure(figsize=(15,20))
    for i,col in enumerate(col_categorical[1:-1],start=1):
        plt.subplot(5,2,i)
        sns.countplot(data[col])
        plt.xlabel(col, fontweight="bold")
    plt.show()
    
    # average price of each make
    plt.figure(figsize=(15,8))
    data.groupby("make")["price"].mean().sort_values(ascending=False).plot.bar()
    plt.xticks(rotation=90)
    plt.xlabel("Make", fontweight="bold")
    plt.ylabel("Count", fontweight="bold")
    plt.title("Countplot of Car Make", fontweight="bold")
    plt.show()
    
    #Heatmap
    plt.figure(figsize=(12,8))
    sns.heatmap(data.corr(), annot=True, cmap="RdYlGn", square=True, mask=np.triu(data.corr(), k=1))
    plt.show()
    
FilePath=input("Enter the path of the file :")
data=load_data(FilePath)

# Correction of typo errors
data["make"] = data['CarName'].str.split(' ', expand=True)[0]
data["make"] = data["make"].replace({"maxda":"mazda", "Nissan":"nissan", "porcshce":"porsche",
                                     "toyouta":"toyota", "vokswagen":"volkswagen", "vw":"volkswagen"})
Requirement=input("Need the imputed data [y/n] :")
if(Requirement=="y"):
    SavingFile=input("Enter the path to save the data(with name and .csv as an extension):")
    data.to_csv(SavingFile)
Visualization(data)