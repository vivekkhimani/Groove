'''
This script is used for training the model based on Spotify dataset and extract the following:
    a) Classes and Euclidean Distances for each sample in X
    b) Trained sklearn model

When new inputs are received from the users, following will take place:
    a) A class will be determined using sklearn model
    b) Random recommendations made with Euclidean Distance < THRESHOLD
'''
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler, Normalizer

import sys, os, csv, json
import itertools
from collections import defaultdict

#import data for generalized recommendations
def import_data(path):
    df = pd.read_csv(path)
    scaler = StandardScaler()
    normalizer = Normalizer()
    #processing
    # df['acousticness'] = normalizer.fit_transform(scaler.fit_transform(df[['acousticness']]))
    # df['danceability'] = normalizer.fit_transform(scaler.fit_transform(df[['danceability']]))
    # df['popularity'] = normalizer.fit_transform(scaler.fit_transform(df[['popularity']]))
    # df['loudness'] = normalizer.fit_transform(scaler.fit_transform(df[['loudness']]))
    # df['tempo'] = normalizer.fit_transform(scaler.fit_transform(df[['tempo']]))
    # df['energy'] = normalizer.fit_transform(scaler.fit_transform(df[['energy']]))
    # df['instrumentalness'] = normalizer.fit_transform(scaler.fit_transform(df[['instrumentalness']]))
    # df['valence'] = normalizer.fit_transform(scaler.fit_transform(df[['valence']]))
    # df['liveness'] = normalizer.fit_transform(scaler.fit_transform(df[['liveness']]))
    # df['speechiness'] = normalizer.fit_transform(scaler.fit_transform(df[['speechiness']]))

    #logs
    # print(df['acousticness'].min(), df['acousticness'].max())
    # print(df['danceability'].min(), df['danceability'].max())
    # print(df['popularity'].min(), df['popularity'].max())
    # print(df['loudness'].min(), df['loudness'].max())
    # print(df['tempo'].min(), df['tempo'].max())
    # print(df['energy'].min(), df['energy'].max())
    # print(df['instrumentalness'].min(), df['instrumentalness'].max())
    # print(df['valence'].min(), df['valence'].max())
    # print(df['liveness'].min(), df['liveness'].max())
    # print(df['speechiness'].min(), df['speechiness'].max())

    #generate X
    X = df[['acousticness', 'danceability', 'popularity',  'loudness', 'tempo', 'energy', 'instrumentalness', 'valence','liveness', 'speechiness']].values
    X = scaler.fit_transform(X)
    return X

def explore_clusters(data):
    inertia_vals = defaultdict()
    silhouette_avg = defaultdict()
    for num_clusters in range(1000, 1001, 250):
        print("Running with num_clusters=", str(num_clusters))
        kmeans = KMeans(n_clusters=num_clusters, n_init=50, max_iter=1000, precompute_distances=True, verbose=0)
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


#driver
if __name__ == '__main__':
    #prepare data
    X = import_data('./data/data.csv')
    #explore hyperparameters
    explore_clusters(X)
    #use the best one to create final clusters
