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
    convert the top user songs retrieved from Spotify into time-series format for TSLearn
    '''
    return

def query_clusters(vectorized_data):
    '''
    Find the matching clusters based on the vectorized user data
    '''
    return

def make_recommendations(matches):
    '''
    Sort the top matches and make top 10-20 recommendations
    '''
    return

#public endpoints
@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/redirect')
def redirect():
    return render_template('home.html')

#driver
port = int(os.environ.get('PORT', 8080))
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=port)
