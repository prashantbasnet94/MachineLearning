# Data Preprocessing Template

# Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading the dataset
dataset = pd.read_csv('Data.csv');

#setting it apart
X = dataset.iloc[:,:-1].values;
Y = dataset.iloc[:,3].values;


#need to do Imputer first
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#encoding all those missing NaN datas
from sklearn.preprocessing import LabelEncoder
labelEncoder_X= LabelEncoder();
X[:,0]=labelEncoder_X.fit_transform(X[:,0])
 
#labelEncoding for purchase as well
from sklearn.preprocessing import LabelEncoder
labelEncoder_Y = LabelEncoder()
Y= labelEncoder_Y.fit_transform(Y);

#since countries named are broken down into 0,1,2, in eqth it matters the value

from sklearn.preprocessing import OneHotEncoder
oneHotEncoder = OneHotEncoder(categorical_features=[0])
X= oneHotEncoder.fit_transform(X).toarray()

#spitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0);

 