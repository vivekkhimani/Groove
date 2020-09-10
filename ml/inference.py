'''
A script for inference which will be deployed on the server. We expect the following files to be present:
	a) {X-index: trackID} - JSON file (will be used to look up track IDs based on filtered indices)
	b) {cluster: [X-indices]} - JSON file (will be used to recommend songs from relevant clusters based on the preferences)
	c) saved sklearn kmeans model - pickle file (will be used to determine classes of input audio features)
	d) {cluster: centroid} - JSON file (will be used to display centroid information on the website)
'''
import numpy as np 
import pandas as pd
from joblib import dump, load
import sys, os, csv, json
from collections import defaultdict

#sklearn
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import mean_squared_error
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE