# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 

#import dataset to var
dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:,:-1].values;
Y = dataset.iloc[:,1].values;

#putting in train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=1/3, random_state=0);


#import linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

Y_predict = regressor.predict(X_test)

#visualizing the training set
#this is the scatter dots

plt.scatter(X_train,Y_train,color='red');
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()


plt.scatter(X_test,Y_test,color='red');
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
