import pandas as pd 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import utils
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

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
    Impute=IterativeImputer(max_iter=10, random_state=610) # Iterative imputation
    Data=pd.DataFrame(Impute.fit_transform(pd.DataFrame(imputed_data)),columns=col_numeric)
    print("Imputation completed....")
    data[col_numeric]=Data[col_numeric]
    return data

def Visualization(data,col_categorical):
    
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
    plt.title("Heatmap")
    sns.heatmap(data.corr(), annot=True, cmap="RdYlGn", square=True, mask=np.triu(data.corr(), k=1))
    plt.show()
    
    #Outliers 
    plt.figure(figsize=(14,7))
    sns.boxplot(data["price"])
    plt.title("Boxplot for outliers detection", fontweight="bold")
    
def Label_Encoder(data,col_categorical):
    encoder=LabelEncoder()
    for col in col_categorical:
        data[col]=encoder.fit_transform(data[col])
    return data        

def Model(data):
    
    X=data.drop(columns="price")
    Y=data["price"]
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=601)
    Dtree=DecisionTreeClassifier(random_state=7)
    Dtree.fit(X_train,Y_train)
    Y_pred=Dtree.predict(X_test)
    print("R-squared (accuracy):", r2_score(Y_pred, Y_test))


FilePath="./Data/CarPrice.csv"
data=load_data(FilePath)

# Correction of typo errors & removing the unwanted columns
data["make"] = data['CarName'].str.split(' ', expand=True)[0]
data["make"] = data["make"].replace({"maxda":"mazda", "Nissan":"nissan", "porcshce":"porsche",
                                     "toyouta":"toyota", "vokswagen":"volkswagen", "vw":"volkswagen"})
data.drop(columns=["CarName","car_ID"],inplace=True)
data["price"]=data["price"].astype(int)
col_categorical = list(data.select_dtypes(include="object")) # Columns with numerical datatype

#Imputed datas
Requirement=input("Need the imputed data [y/n] :")
if(Requirement=="y"):
    SavingFile=input("Enter the path to save the data(with name and .csv as an extension):")
    data.to_csv(SavingFile)

Visualization(data,col_categorical)
Label_Encoder(data,col_categorical)
Model(data)
