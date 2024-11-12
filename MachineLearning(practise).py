import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

dataset = pd.read_csv(r"E:\Data.csv")

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:,3].values

from sklearn.impute import SimpleImputer #spyder 4

imputer = SimpleImputer()

imputer = imputer.fit(x[:,1:3])

x[:, 1:3] = imputer.transform(x[:,1:3])

from sklearn.preprocessing import LabelEncoder

labelencoder_x = LabelEncoder()

labelencoder_x.fit_transform(x[:,0])

labelencoder_y  = LabelEncoder()

labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size= 0.2, random_state=0)

#feature scaling 

from sklearn.preprocessing import Normalizer

sc_x = Normalizer()

x_train = sc_x.fit_transform(x_train)

x_test = sc_x.transform(x_test)
