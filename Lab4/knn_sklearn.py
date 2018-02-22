#import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

#read data file as dataframe
data = pd.read_csv('iris.csv')

#take four numeric features as X input
X = data.values[:, :4]

#create an array of length 150 named y
y = np.zeros(150)

#encoding classes to numbers
for i in range(len(y)):
    if data.values[i, 4]=='setosa':
        y[i] = 0
    elif data.values[i, 4]=='versicolor':
        y[i] = 1
    elif data.values[i, 4]=='virginica':
        y[i] = 2

#randomly shuffle the whole dataset and create train-test partition
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#create KNN Model
KNN_model = KNeighborsClassifier(n_neighbors=3)

#train the model to fit parameters
KNN_model.fit(X_train, y_train)

#predict y(class) values of all X_test values
y_predict = KNN_model.predict(X_test)

#accuracy calculation
accuracy_score(y_test, y_predict)