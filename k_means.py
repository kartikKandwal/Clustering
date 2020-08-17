'''
1.importing library
2.importing dataset
3.using elbow method to find optimal no. of cluster
4.training the k_means model on the dataset
5.visualising the clustering
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


d1=pd.read_csv('Mall_Customers.csv')
X=d1.iloc[:,[3,4]].values

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('elbow method')
plt.xlabel("no. of cluster")
plt.ylabel('wcss')
plt.show()

kmeans=KMeans(n_clusters=2,init='k-means++',random_state=42)
y_kmeans=kmeans.fit_predict(X)
print(y_kmeans)

plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='cluster1')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label='cluster2')
# plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='cluster3')
# plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='cyan',label='cluster4')
# plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='magenta',label='cluster5')
# plt.scatter(X[y_kmeans==5,0],X[y_kmeans==5,1],s=100,c='pink',label='cluster6')
# # plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:1],s=300,c='yellow',label='centroid')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('cluster of customer')
plt.xlabel("annual income")
plt.ylabel('spend score (1-100)')
plt.show()