# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values



#read dataset of csv
dataset = pd.read_csv('Data.csv')

#store all values from all row and column expect for the last cloumn
x = dataset.iloc[:,:-1].values
#store all values from  all row but from only third cloumn
y = dataset.iloc[:,3].values

 
#taking care of missing data

from sklearn.preprocessing import Imputer
imputer  = Imputer(missing_values='NaN',strategy = 'mean', axis =0)
imputer  = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])


#Taking care of missing data
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
x[:,0]=labelEncoder.fit_transform(x[:,0])