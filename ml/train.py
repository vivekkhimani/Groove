'''
This script is used for training the model based on Spotify dataset and extract the following:
    a) Classes and Euclidean Distances for each sample in X
    b) Trained sklearn model

When new inputs are received from the users, following will take place:
    a) A class will be determined using sklearn model
    b) Random recommendations made with Euclidean Distance < THRESHOLD

For the clusters, we generate a visualization plots using the following strategies:
    a) PCA
    b) TSNE
'''
#general
import numpy as np
import pandas as pd
from joblib import dump, load
import sys, os, csv, json
import itertools
from collections import defaultdict

#sklearn
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import mean_squared_error
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#plotly
import plotly as py 
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

#import data for generalized recommendations
def import_data(path):
    df = pd.read_csv(path)
    scaler = StandardScaler()
    normalizer = Normalizer()

    #generate X
    X = df[['acousticness', 'danceability', 'popularity',  'loudness', 'tempo', 'energy', 'instrumentalness', 'valence','liveness', 'speechiness']].values
    X = scaler.fit_transform(X)
    #logs
    print(X) 
    return X

def explore_clusters(data):
    inertia_vals = defaultdict()
    silhouette_avg = defaultdict()
    for num_clusters in range(1000, 1100, 250):
        print("Running with num_clusters=", str(num_clusters))
        kmeans = KMeans(n_clusters=num_clusters, n_init=50, max_iter=1000, verbose=1)
        predictions = kmeans.fit_predict(data)
        sil_avg = silhouette_score(data, predictions)
        silhouette_avg[num_clusters] = sil_avg
        inertia_vals[num_clusters] = kmeans.inertia_
    print(sil_avg)
    print(inertia_vals)

def create_clusters(data, num_clusters):
    clustered_tally = defaultdict(list)
    kmeans = KMeans(n_clusters=num_clusters, n_init=50, max_iter=1000, verbose=1)
    predictions = kmeans.fit_predict(data)
    sil_score = silhouette_score(data, predictions)
    inertia = kmeans.inertia_
    for indices, items in enumerate(predictions):
        clustered_tally[items].append(indices)
    clustered_tally = dict(clustered_tally)

    #save the clustered tally
    with open('cluster.json', 'w') as fp:
        json.dump(clustered_tally, fp)

    #save the sklearn model
    joblib.dump(kmeans, "saved_kmeans.pkl")

def visualize_PCA(X, predictions):
    return

def visualize_TSNE(X, predictions):
    return

#driver
if __name__ == '__main__':
    #prepare data
    X = import_data('./data/data.csv')
    #explore hyperparameters
    explore_clusters(X)
    #use the best one to create final clusters
