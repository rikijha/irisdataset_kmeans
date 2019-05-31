# -*- coding: utf-8 -*-
"""
Created on Fri May 31 17:41:03 2019

@author: Dell
"""

import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Iris.csv')

X=dataset.iloc[:,1:5].values
wcss=[] #width cluster sum of squares

from sklearn.cluster import KMeans
for i in range(1,10):
    k_means=KMeans(n_clusters=i,random_state=0)
    k_means.fit(X)
    wcss.append(k_means.inertia_)


#Elbow plot
plt.plot(range(1,10),wcss)
plt.title('Elbow plot')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()

k_means=KMeans(n_clusters=2,random_state=0)
y=k_means.fit_predict(X)

plt.scatter(X[y==0,0],X[y==0,1],c='blue',label='Iris-versicolor/Iris-virginica')
plt.scatter(X[y==1,0],X[y==1,1],c='red',label='Iris-setosa')
plt.scatter(k_means.cluster_centers_[:,0],k_means.cluster_centers_[:,1],c='yellow',
            label='centroid',s=200)
plt.legend()