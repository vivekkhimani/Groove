from flask import Flask, request, jsonify, Response, send_from_directory, url_for, render_template
import requests, json
import ast
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
    - Find the top-10 nearest songs based on the inertia score
    - Return id's
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

def main():
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

@app.route('/top_songs')
def top_songs():
    return

@app.route('/playlist')
def playlist():
    return

@app.route('/recently_played')
def recently_played():
    return

@app.route('saved_tracks')
def saved_tracks():
    return

#driver
port = int(os.environ.get('PORT', 8080))
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=port)
