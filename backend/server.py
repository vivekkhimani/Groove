from flask import Flask, request, jsonify, Response, send_from_directory, url_for, render_template, make_response
import requests, json
import ast
import time, random
import sys, os, csv, json
import numpy as np
import pandas as pd
import joblib
from collections import defaultdict

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import plotly.express as px

#flask init
app = Flask(__name__)
SERVER_ROOT = os.path.dirname(__file__)

#private methods
def parse_user_favorites(access_token, id_list):
    scaler = StandardScaler()
    headers = {'Authorization': 'Bearer ' + access_token}
    ids_string = ''
    for indices, items in enumerate(id_list):
        ids_string+=str(items)
        limit = len(id_list) - 1
        if indices < limit:
            ids_string+='%2C'
    full_link = 'https://api.spotify.com/v1/audio-features?ids='+ids_string
    audio_features = requests.get(full_link, headers=headers)
    audio_json = audio_features.json()
    df = pd.DataFrame(audio_json['audio_features'])
    required_features = ['acousticness', 'danceability',  'loudness', 'tempo', 'energy', 'instrumentalness', 'valence','liveness', 'speechiness']
    X = df[required_features].values
    X = pd.DataFrame(scaler.fit_transform(X))
    X.columns = required_features
    return X

def query_clusters(data):
    pca = PCA(n_components=2)
    compressed_data = pca.fit_transform(data)
    model_url = os.path.join(SERVER_ROOT, 'inference', 'saved_kmeans_compressed_2_clusters_1000.pkl')
    kmeans = joblib.load(open(model_url, 'rb'))
    predictions = kmeans.predict(compressed_data)

    #send data for visualization
    send_df = pd.DataFrame(compressed_data)
    col_names = list()
    for i in range(2):
        col_names.append('PCA_'+str(i))
    send_df.columns = col_names
    int_pred = predictions.astype(int)
    send_df['cluster'] = int_pred
    img_url = make_visualizations(send_df, 2, 1000) 
    return predictions, img_url

def make_recommendations(access_token, id_list, predictions, num_recommendations):
    name_list = list()
    base_tracks_url = list()
    headers = {'Authorization': 'Bearer ' + access_token}
    ids_string = ''
    for indices, items in enumerate(id_list):
        ids_string+=str(items)
        limit = len(id_list) - 1
        if indices < limit:
            ids_string+='%2C'
    full_link = 'https://api.spotify.com/v1/tracks?ids='+ids_string+"&market=US"
    track_details = requests.get(full_link, headers=headers)
    track_json = track_details.json()
    for items in track_json['tracks']:
        name_list.append(items['name'])
        base_tracks_url.append(items['external_urls']['spotify'])

    saved_cluster_path = os.path.join(SERVER_ROOT, 'inference', 'saved_cluster_index_compressed_2_clusters_1000.json')
    with open(saved_cluster_path) as f:
        saved_cluster_data = json.load(f)

    saved_id_path = os.path.join(SERVER_ROOT, 'inference', 'index-id.json')
    with open(saved_id_path) as f:
        saved_id_references = json.load(f)

    saved_name_path = os.path.join(SERVER_ROOT, 'inference', 'id-name.json')
    with open(saved_name_path) as f:
        saved_name_references = json.load(f)

    saved_centroid_path = os.path.join(SERVER_ROOT, 'inference', 'saved_cluster_centroid_compressed_2_clusters_1000.json')
    with open(saved_centroid_path) as f:
        saved_centroid_references = json.load(f)

    final_predicted_ids = list()
    predicted_tracks = list()
    reference = 0
    while len(final_predicted_ids) < num_recommendations:
        if reference == len(predictions):
            reference = 0
        selected_cluster = predictions[reference]
        random_index = random.choice(saved_cluster_data[str(selected_cluster)])
        track_id = saved_id_references[str(random_index)]
        track_name = saved_name_references[track_id]
        link = 'https://api.spotify.com/v1/tracks?ids='+track_id+"&market=US"
        track_det = requests.get(link, headers=headers)
        track_js = track_det.json()
        track_url = track_js['tracks'][0]['external_urls']['spotify']
        if (track_id not in id_list) and (track_id not in final_predicted_ids):
            append_dict = {"recommended_track_id":track_id, "recommended_track_name":track_name, "recommended_track_url":track_url, "base_track_id":id_list[reference], "base_track_name":name_list[reference], "base_track_url":base_tracks_url[reference], "associated_cluster":str(selected_cluster), "cluster_centroid":saved_centroid_references[str(selected_cluster)]}
            final_predicted_ids.append(append_dict)
            predicted_tracks.append(track_id)
        reference+=1
        continue
    return final_predicted_ids, predicted_tracks

def make_visualizations(X, reduce_comp, num_clusters):
    overall_data_path = os.path.join(SERVER_ROOT, 'inference', 'all_compressed_data.csv')
    overall_X = pd.read_csv(overall_data_path)

    saved_centroid_path = os.path.join(SERVER_ROOT, 'inference', 'saved_cluster_centroid_compressed_2_clusters_1000.json')
    with open(saved_centroid_path) as f:
        saved_centroid_references = json.load(f)

    req_clusters = X.cluster.unique().tolist()
    filtered_X = overall_X[overall_X['cluster'].isin(req_clusters)]
    filtered_X = filtered_X.rename(columns={'0':'PCA_0', '1':'PCA_1'})
    centroid_ref = list()
    for items in X['cluster']:
        rounded_ref = list(np.around(np.array(saved_centroid_references[str(items)]), 2))
        centroid_ref.append(rounded_ref)
    X['centroids'] = centroid_ref
    fig = px.scatter(filtered_X, x='PCA_0', y='PCA_1', color='cluster')
    full_title = "Recommended Clusters: original_dimensions=9, PCA_dimensions="+str(reduce_comp)
    fig.update_layout(
        title=full_title)
    plot_name = "plot_" + str(time.time()) + ".svg"
    print(plot_name)
    full_svg_path = os.path.join(SERVER_ROOT, 'plots', plot_name)
    fig.write_image(full_svg_path)
    img_url = 'https://c74b0f401e15.ngrok.io/plots/' + plot_name
    return img_url

def main_ml(access_token, id_list, num_recommendations):
    audio_features = parse_user_favorites(access_token, id_list)
    predictions, img_url = query_clusters(audio_features)
    recommended_dict, recommended_ids = make_recommendations(access_token, id_list, predictions, num_recommendations)
    return {"recommendations":recommended_dict, "plot":img_url}

#public endpoints
@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/redirect')
def redirect():
    return render_template('home.html')

@app.route('/plots/<path:path>', methods=['GET'])
def get_plots(path):
    full_path = "./plots/" + path
    resp = make_response(open(full_path).read())
    resp.content_type = 'image/svg+xml'
    return resp

@app.route('/get_user_playlist')
def get_playlist():
    global user_playlist
    user_playlist = list()
    access_token = request.args.get("access_token")
    headers = {'Authorization': 'Bearer ' + access_token}
    playlist_data = requests.get('https://api.spotify.com/v1/me/playlists', headers=headers)
    playlist_json = playlist_data.json()
    return_list = list()
    for items in playlist_json['items']:
        return_list.append([items['id'], items['name']])
    return jsonify(return_list)

@app.route('/top_songs')
def top_songs():
    access_token = request.args.get("access_token")
    headers = {'Authorization': 'Bearer ' + access_token}
    top_songs_data = requests.get(' https://api.spotify.com/v1/me/top/tracks?time_range=medium_term&limit=20&offset=5', headers=headers)
    top_songs_json = top_songs_data.json()
    tracks_list = top_songs_json['items']
    id_list = list()
    for items in tracks_list:
        id_list.append(items['id'])
    rec_dict = main_ml(access_token, id_list, 10)
    return rec_dict

@app.route('/playlist')
def playlist():
    access_token = request.args.get("access_token")
    playlist_id = request.args.get("req_id")
    headers = {'Authorization': 'Bearer ' + access_token}
    full_link = "https://api.spotify.com/v1/playlists/"+playlist_id+"/tracks?market=US&fields=items(added_by.id%2Ctrack(name%2Cid%2Chref%2Calbum(name%2Chref)))&limit=50&offset=5";
    playlist_data = requests.get(full_link, headers=headers);
    playlist_json = playlist_data.json()
    id_list = list()
    for items in playlist_json['items']:
        id_list.append(items['track']['id'])
    rec_dicts = main_ml(access_token, id_list, 10)
    return rec_dicts

@app.route('/recently_played')
def recently_played():
    access_token = request.args.get("access_token")
    headers = {'Authorization': 'Bearer ' + access_token}
    current_time = time.time()
    deduct_time = 8.64e+7
    after_stamp = int(current_time - deduct_time)
    spotify_req_url = 'https://api.spotify.com/v1/me/player/recently-played?type=track&limit=10&after='+str(after_stamp)
    recently_played_data = requests.get(spotify_req_url, headers=headers)
    recently_played_json = recently_played_data.json()
    tracks_list = recently_played_json['items']
    id_list = list()
    for items in tracks_list:
        id_list.append(items['track']['id'])
    rec_dict = main_ml(access_token, id_list, 10)
    return rec_dict

@app.route('/saved_tracks')
def saved_tracks():
    access_token = request.args.get("access_token")
    headers = {'Authorization': 'Bearer ' + access_token}
    saved_tracks_data = requests.get('https://api.spotify.com/v1/me/tracks?market=US&limit=20&offset=5', headers=headers)
    saved_tracks_json = saved_tracks_data.json()
    tracks_list = saved_tracks_json['items']
    id_list = list()
    for items in  tracks_list:
        id_list.append(items['track']['id'])
    rec_dict = main_ml(access_token, id_list, 10)
    return rec_dict

#driver
port = int(os.environ.get('PORT', 8080))
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=port)
