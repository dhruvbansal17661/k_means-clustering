# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 12:04:07 2019

@author: Acer
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]]

import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X , method = 'ward'))
plt.show() 

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
Y_hc = hc.fit_predict(X)

plt.scatter(X[Y_hc == 0 , 0] , X[Y_hc == 0, 1] , s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[Y_hc == 1 , 0] , X[Y_hc == 1, 1] , s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[Y_hc == 2 , 0] , X[Y_hc == 2, 1] , s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[Y_hc == 3 , 0] , X[Y_hc == 3, 1] , s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[Y_hc == 4 , 0] , X[Y_hc == 4, 1] , s = 100, c = 'magenta', label = 'Cluster 5')
plt.legend()
plt.show()