'''
This script is used for training the model based on Spotify dataset and extract the following:
    a) Classes and Euclidean Distances for each sample in X
    b) Trained sklearn model

When new inputs are received from the users, following will take place:
    a) A class will be determined using sklearn model
    b) Random recommendations made with Euclidean Distance < THRESHOLD

For the clusters, we generate a visualization plots using the following strategies:
    a) PCA
'''
#general
import numpy as np
import pandas as pd
import joblib
import sys, os, csv, json
from collections import defaultdict

#sklearn
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import mean_squared_error
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#plotly
import plotly.express as px

#import data for generalized recommendations
def import_data(path):
    df = pd.read_csv(path)
    scaler = StandardScaler()
    normalizer = Normalizer()
    selected_features = ['acousticness', 'danceability',  'loudness', 'tempo', 'energy', 'instrumentalness', 'valence','liveness', 'speechiness']
    #generate X
    X = df[selected_features].values
    X = pd.DataFrame(scaler.fit_transform(X))
    X.columns = selected_features

    #save index-id references for inference
    id_s = df['id']
    id_dict = id_s.to_dict()
    full_path = './inference_req/index-id.json'
    if not os.path.exists(full_path):
        with open('./inference_req/index-id.json', 'w+') as fp:
            json.dump(id_dict, fp)
    return X

def explore_clusters(data, reduce_comp, num_clusters):
    #init
    inertia_vals = defaultdict()
    silhouette_avg = defaultdict()
    pca = PCA(n_components=reduce_comp)
    comp_data = pca.fit_transform(data)

    #training
    print("Training with num clusters = ", num_clusters)
    full_path = './inference_req/saved_kmeans_compressed_'+str(reduce_comp)+'_clusters_'+str(num_clusters)+'.pkl'
    if os.path.exists(full_path):
        kmeans = joblib.load(open(full_path, 'rb'))
        predictions = kmeans.predict(comp_data)
    else:
        kmeans = KMeans(n_clusters=num_clusters, n_init=10, max_iter=300, verbose=1)
        predictions = kmeans.fit_predict(comp_data)
    sil_avg = silhouette_score(comp_data, predictions)
    silhouette_avg[num_clusters] = sil_avg
    inertia_vals[num_clusters] = kmeans.inertia_
    print(sil_avg)
    print(inertia_vals)

    #save sklearn model
    full_path = './inference_req/saved_kmeans_compressed_'+str(reduce_comp)+'_clusters_'+str(num_clusters)+'.pkl'
    joblib.dump(kmeans, full_path)

    #save cluster-index references
    clustered_index_tally = dict()
    predictions = predictions.astype(int)
    for indices, items in enumerate(predictions):
        if items not in clustered_index_tally:
            clustered_index_tally[int(items)] = [indices]
        else:
            clustered_index_tally[int(items)].append(indices)
    clustered_index_tally = dict(clustered_index_tally)
    full_path = './inference_req/saved_cluster_index_compressed_'+str(reduce_comp)+'_clusters_'+str(num_clusters)+'.json'
    with open(full_path, 'w+') as fp:
        json.dump(clustered_index_tally, fp)

    #save cluster-centroid references
    cluster_centroid_tally = dict()
    for indices, items in enumerate(kmeans.cluster_centers_):
        cluster_centroid_tally[indices] = items.tolist()
    full_path = './inference_req/saved_cluster_centroid_compressed_'+str(reduce_comp)+'_clusters_'+str(num_clusters)+'.json'
    with open(full_path, 'w+') as fp:
        json.dump(cluster_centroid_tally, fp)

    compressed_data = pd.DataFrame(comp_data)
    col_names = list()
    for i in range(reduce_comp):
        col_names.append('PCA_'+str(i))
    compressed_data.columns = col_names
    predictions = predictions.astype(int)
    compressed_data["cluster"] = predictions
    visualize_PCA(compressed_data, reduce_comp, num_clusters)


def visualize_PCA(X, reduce_comp, num_clusters):
    if reduce_comp == 3:
        fig = px.scatter_3d(X, x='PCA_0', y='PCA_1', z='PCA_2', color='cluster')
        full_title = "lib_size=170000 songs, num_clusters="+str(num_clusters)+", original_dim=9, PCA_dim="+str(reduce_comp)
        fig.update_layout(
            title=full_title)
        full_png_path = './plots/overall_clustering_3dplot_compressed_'+str(reduce_comp)+'_clusters_'+str(num_clusters)+'.png'
        full_svg_path = './plots/overall_clustering_3dplot_compressed_'+str(reduce_comp)+'_clusters_'+str(num_clusters)+'.svg'
        fig.write_image(full_png_path)
        fig.write_image(full_svg_path)

    elif reduce_comp == 2:
        fig = px.scatter(X, x='PCA_0', y='PCA_1', color='cluster')
        full_title = "lib_size=170000 songs, num_clusters="+str(num_clusters)+", original_dim=9, PCA_dim="+str(reduce_comp)
        fig.update_layout(
            title=full_title,
            autosize=False,
            width=1000,
            height=800)
        full_png_path = './plots/overall_clustering_2dplot_compressed_'+str(reduce_comp)+'_clusters_'+str(num_clusters)+'.png'
        full_svg_path = './plots/overall_clustering_2dplot_compressed_'+str(reduce_comp)+'_clusters_'+str(num_clusters)+'.svg'
        fig.write_image(full_png_path)
        fig.write_image(full_svg_path)

#driver
if __name__ == '__main__':
    #prepare data
    X = import_data('./data/data.csv')
    #explore hyperparameters
    explore_clusters(X, 2, 1000)
    #use the best one to create final clusters
