import pandas as pd 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def load_data(path):
    data=pd.read_csv(path)
    #print(data.info())
    missing_val=(data.isna().sum()) #Counts of missing values before imputation
    #print(missing_val)
    if (sum(missing_val)>0):
        Data=Imputation(data,missing_val)
    missing_val=(data.isna().sum()) # Counts of missing values after imputation 
    #print(missing_val)
    return Data

def Imputation(data,missing_val):
    print("Impuation started....")
    col_numeric = list(data.select_dtypes(exclude="object")) # Columns with numerical datatype
    col_categorical = list(data.select_dtypes(include="object"))#Columns with Categorical datatype
    imput_data=pd.DataFrame(data,columns=col_numeric)
    Impute=IterativeImputer(max_iter=10, random_state=610)
    Data=pd.DataFrame(Impute.fit_transform(pd.DataFrame(imput_data)),columns=col_numeric)
    print("Imputation completed....")
    data[col_numeric]=Data[col_numeric]
    return data

FilePath=load_data(input("Enter the path of the file :"))
print(pd.DataFrame(FilePath))

#data["make"] = data['CarName'].str.split(' ', expand=True)[0]
#data["make"] = data["make"].replace({"maxda":"mazda","Nissan":"nissan","porcshce":"porsche",
 #                                   "toyouta":"toyota","vokswagen":"volkswagen","vw":"volkswagen"})
#print(data["make"].unique())