from flask import Flask, request, jsonify, Response, send_from_directory, url_for, render_template
import requests, json
import ast
import time
import sys, os, csv, json
import numpy as np
import pandas as pd
from collections import defaultdict

#flask init
app = Flask(__name__)

#private methods
def parse_user_favorites(user_fav):
    '''
    - Get user preferred songs using Spotify API.
    - Call the Spotify API and get audio features for each of the songs
    - Convert the audio features in a pandas dataframe and return
    '''
    return

def query_clusters(vectorized_data):
    '''
    - Import the saved sklearn trained model
    - Get the designated cluster using the sklearn model
    - From each of the designated clusters, randomly pick the "closest" song such that it's not already picked before
    '''
    return

def make_recommendations(matches):
    '''
    - Based on the retrieved id's, get track names, and display it on the interface
    '''
    return

def make_visualizations(data):
    '''
    - Generate plots using the recommendations and display it on the frontend
    '''
    return

def main(user_data):
    '''
    - Wrapper for all the aforementioned private methods
    '''
    return

#public endpoints
@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/redirect')
def redirect():
    return render_template('home.html')

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
    #FIXME: Insert the ML part here
    return jsonify(id_list)

@app.route('/playlist')
def playlist():
    #FIXME: need to fix the listener issue on client side
    access_token = request.args.get("access_token")
    req_id = request.args.get("req_id")
    return jsonify(req_id)

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
    #FIXME: Insert the ML part here
    return jsonify(id_list)

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
    #FIXME: Insert the ML part here
    return jsonify(id_list)

#driver
port = int(os.environ.get('PORT', 8080))
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=port)
