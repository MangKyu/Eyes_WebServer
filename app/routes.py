
from flask import render_template, request, datetime, jsonify
from app import app
import os

PROFILE_FOLDER = os.path.join('static', 'Image')
app.config['UPLOAD_FOLDER'] = PROFILE_FOLDER

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

@app.route("/sendImage", methods=['POST'])
def save_image():
    mirror_uid = request.values.get('mirrorUid')
    uid = request.values.get('uid')
    file = request.files.get('Image')

    file_ext = os.path.splitext(file.filename)[1]
    file_name = uid + '_' + datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S') + file_ext

    try:
        file_dir = os.path.join(app.config['IMAGE_FOLDER'], mirror_uid, 'user_photos', uid)
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir)

    except Exception as e:
        print(e)
    finally:
        db.insertUser()
        file_path = file_dir + '//' + file_name
        file.save(file_path)


@app.route("/getPath", methods=['GET'])
def send_path():
    startX = request.args.get('startX')
    startY = request.args.get('startY')
    endX = request.args.get('endX')
    endY = request.args.get('endY')
    return render_template('path.html',startX=startX, startY=startY, endX=endX, endY=endY)