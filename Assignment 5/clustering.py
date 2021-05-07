#-------------------------------------------------------------------------------------------------------
# AUTHOR: Dai (Paul) Vuong
# FILENAME: clustering.py
# SPECIFICATION: run k-means multiple times and check which k value maximizes the Silhouette coefficient
# FOR: CS 4200- Assignment #5
# TIME SPENT: 2 hours
#------------------------------------------------------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics
from operator import itemgetter

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix
X_training = np.array(df.values)

#run kmeans testing different k values from 2 until 20 clusters
silhouette_scores = dict()
for k in range(2,21):
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     kmeans = KMeans(n_clusters=k, random_state=0)
     kmeans.fit(X_training)

     #for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
     #find which k maximizes the silhouette_coefficient
     silhouette_coeff = silhouette_score(X_training, kmeans.labels_)
     silhouette_scores[k] = (silhouette_coeff)
     

#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
k_value = [x for x in range(2,21)]
plt.plot(silhouette_scores.keys(),silhouette_scores.values(),)
# plt.show()
best_k = dict(sorted(silhouette_scores.items(), key = itemgetter(1), reverse = True)[:1])

#reading the validation data (clusters) by using Pandas library
df1 = pd.read_csv('testing.csv', header=None)

#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
labels = np.array(df1.values).reshape(1,-1)[0]

#Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())

#rung agglomerative clustering now by using the best value o k calculated before by kmeans
#Do it:
agg = AgglomerativeClustering(n_clusters=best_k.keys()[0], linkage='ward')
agg.fit(X_training)

# Calculate and print the Homogeneity of this agglomerative clustering
print("Agglomerative Clustering Homogeneity Score = " + metrics.homogeneity_score(labels, agg.labels_).__str__())
