# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from sklearn import datasets
import numpy as np
import math
from sklearn.model_selection import train_test_split


iris = datasets.load_iris()
X = iris.data[:, :2] 
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33,random_state=42)

train_set_size = len(y_train)
test_set_size = len(y_test)

distances = np.zeros((1,train_set_size))


for i in range(test_set_size) :
    x_1 = X_test[i,:]
    for j in range(train_set_size) :
        x_2 = X_train[j,:]
        #get euclidean distance between x_1 and x_2
        distance[0, j] = math.sqrt( (np.sum((x_1-x_2)**2) )
        
    #find nearest neighbors
    num_of_neighbours = 3
    neighbours = distance[0, :].argsort()[:num_of_neighbours]
    
    #find out the majorioty class
    num_of_classes = len(np.unique(y_train))
    majority_class = np.zeroes(1, num_of_classes)
    
    for j in range(num_of_neighbours) :
        majority_class[0,y_train[neighbours[j]]] += 1 
    
    #perform classification here
    predictions = np.zeros((test_set_size,1))
    predicted_class = np.argmax(majority_class[0,:])
    predictions[i] = predicted_class    

