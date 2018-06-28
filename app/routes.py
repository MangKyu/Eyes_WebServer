import datetime
from flask import render_template, request, jsonify
from app import app
import os

PROFILE_FOLDER = os.path.join('Files', 'Profile')
FACE_FOLDER = os.path.join('Files', 'FaceImage')
app.config['PROFILE_FOLDER'] = PROFILE_FOLDER
app.config['FACE_FOLDER'] = FACE_FOLDER


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/peerList', methods=['GET'])
def send_peer_list():
    data = {'ip': '1', 'port': '5000', 'file_hash': 'aaa'}
    return jsonify(data)


@app.route("/getPath", methods=['GET'])
def send_path():
    startX = request.args.get('startX')
    startY = request.args.get('startY')
    endX = request.args.get('endX')
    endY = request.args.get('endY')
    return render_template('path.html',startX=startX, startY=startY, endX=endX, endY=endY)