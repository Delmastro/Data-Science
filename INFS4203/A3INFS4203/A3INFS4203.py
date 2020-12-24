from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('data.csv')

X = data[['x','y']].values
# label = data['label'].values

kmeansModel = KMeans(n_clusters=3, random_state = 1)
# kmeansModel = KMeans(n_clusters=4, random_state = 1)


kmeansModel.fit(X)
y_kmeans = kmeansModel.predict(X)

plt.scatter(X[:,0], X[:,1], c=y_kmeans, s=50, cmap='cividis')

# centers = kmeansModel.cluster_centers_
# plt.scatter(centers[:, 0], centers[:,1], c='black', marker='c', s=200, alpha= 0.9)
plt.title('Assignment 3 Dataset')
plt.plot()
plt.show()


# plt.xlim([0, 10])
# plt.ylim([0, 10])

#plt.scatter(x1, x2)


# # create new plot and data
# plt.plot()
# X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
# colors = ['b', 'g', 'r']
# markers = ['o', 'v', 's']

# # k means determine k
# distortions = []
# K = range(1,6)
# for k in K:
#     kmeanModel = KMeans(n_clusters=k).fit(X)
#     kmeanModel.fit(X)
#     distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# # Plot the elbow
# plt.plot(K, distortions, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method showing the optimal k')
# plt.show()

 