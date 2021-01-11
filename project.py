#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# 
# # DATA MINING PROJECT
# 
# by Oyku Dila Akansu and Mats Wapstra

# In[20]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as shc
from toolbox import clusterPlot as cp
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

######################################################
#            Uploading the whole dataset             #
steam = pd.read_csv("steam-200k.csv")
steam.columns = ['User ID', 'Game Names', 'Status', 'Hours', 'E']
steam = steam.drop(columns=['E'], axis=1)

######################################################
#         Grouping, sorting and cutting data         #

#below we cut the rows with purchase and the columns
#User ID and Status
play = steam[~steam.Status.str.contains("purchase")]
play = play.drop(columns=['User ID', 'Status'])
#play_sorted = play.sort_values(by=['Hours'])
#below we group the values by game names
play_grouped = play.groupby(['Game Names']).sum()
#then we sort the values by hour in descending order
play_grouped = play_grouped.sort_values(by=['Hours'], ascending=False)
groupnames = play_grouped.index.values
play_grouped = np.array(play_grouped)
print(play_grouped)

######################################################
#        Using label encoding for game names         #

le = preprocessing.LabelEncoder()
le.fit(groupnames)
game_coded = le.transform(groupnames)
game_trans = np.array(game_coded)
game_trans = list(dict.fromkeys(game_trans))
game_trans = np.array(game_trans)
print(game_trans)

#inverse = list(le.inverse_transform([922, 673, 2994]))
#inverse = np.array(inverse)


# In[9]:


#     Slicing and creating the dataset for use     #
merged = np.column_stack((play_grouped, game_trans))
dataset = merged[30:130]
print(dataset)


# In[10]:


#    Plotting the dataset prepared above    #
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
cluster.fit_predict(dataset)
print(cluster.labels_)
plt.scatter(dataset[:,0], dataset[:,1], c=cluster.labels_, cmap='rainbow')
plt.ylabel('Game Names')
plt.xlabel('Hours')
plt.title('Total Gameplay Hours')


# In[5]:


#     Applying hierarchial clustering     #
plt.figure(figsize=(15, 9))  
dend = shc.dendrogram(shc.linkage(dataset, method='ward'))
plt.ylabel('Games')
plt.xlabel('Hours')
plt.title('Total Gameplay Hours')
plt.show()


# In[6]:


#       Applying K-Means       # 
kmeans = KMeans(n_clusters=3).fit(dataset)
centroids = kmeans.cluster_centers_
#kmeans.predict(dataset)
#kmeans = KMeans(n_clusters=3).fit_transform(dataset)
y_pred = kmeans.predict(dataset)
print(centroids)
cp.clusterPlot(dataset, y_pred, centroids=centroids)
#plt.scatter(dataset[:,0],dataset[:,1], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
#plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=20)
plt.title('Total Gameplay Hours')
plt.xlabel("Game Labels")
plt.ylabel("Total Hours")
plt.show()

#
