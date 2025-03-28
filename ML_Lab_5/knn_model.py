import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,root_mean_squared_error,mean_absolute_percentage_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score 
from sklearn.metrics import calinski_harabasz_score 
from sklearn.metrics import davies_bouldin_score 

def k_means_cluster(X_train):

    kmeans = KMeans(n_clusters=2, random_state=0,n_init="auto").fit(X_train)  
    labels=kmeans.labels_ 
    cluster_labels=kmeans.cluster_centers_ 
    kmeans = KMeans(n_clusters=2, random_state=42).fit(X_train) 
    Sh_scores=silhouette_score(X_train, kmeans.labels_)
    ch_scores=calinski_harabasz_score(X_train, kmeans.labels_) 
    db_indexs=davies_bouldin_score(X_train, kmeans.labels_)

    
    return labels,cluster_labels,Sh_scores,ch_scores,db_indexs

def k_values_plot(X_train):
    sh_score=[]
    ch_score=[]
    db_index=[]
    for k in range(2,12):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X_train) 
        sh_score.append(silhouette_score(X_train, kmeans.labels_))
        ch_score.append(calinski_harabasz_score(X_train, kmeans.labels_))
        db_index.append(davies_bouldin_score(X_train, kmeans.labels_))
        
    plt.figure(figsize=(10,6))
    plt.plot(range(2,12),sh_score,marker="o",linestyle="-",color="blue",label="SH Score")
    plt.plot(range(2,12),ch_score,marker="o",linestyle="-",color="green",label="CH Score")
    plt.plot(range(2,12),db_index,marker="o",linestyle="-",color="red",label="DB Index")
        
    plt.xlabel("No.of Clusters:(k)")
    plt.ylabel("Scores")
    plt.show()

def distortion_plot(X_train):
    distortions=[]
    for k in range(2, 20):
        kmeans = KMeans(n_clusters=k).fit(X_train) 
        distortions.append(kmeans.inertia_) 
    plt.plot(distortions) 
    plt.show()