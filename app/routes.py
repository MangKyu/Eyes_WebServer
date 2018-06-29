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


@app.route("/sendImage", methods=['POST'])
def send_image():
    user_id = request.values.get("userId")
    image_file = request.files.get('Image')

    file_ext = os.path.splitext(image_file.filename)[1]
    file_name = user_id + '_' + datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S') + file_ext

    try:
        file_dir = os.path.join(app.config['PROFILE_FOLDER'], user_id)
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir)

    except Exception as e:
        print(e)
    finally:
        file_path = os.path.join(file_dir, file_name)
        image_file.save(file_path)


@app.route("/getPath", methods=['GET'])
def get_path():
    '''
    start_x = "126.9850380932383"
    start_y = "37.566567545861645"
    end_x = "127.10331814639885"
    end_y = "37.403049076341794"
    '''
    startX = request.args.get('startX')
    startY = request.args.get('startY')
    endX = request.args.get('endX')
    endY = request.args.get('endY')

    return render_template('path.html', startX=startX, startY=startY, endX=endX, endY=endY)


@app.route("/path", methods=['GET'])
def path():
    start_x = request.args.get('startX')
    start_y = request.args.get('startY')
    end_x = request.args.get('endX')
    end_y = request.args.get('endY')

    return render_template('test.html', startX=start_x, startY=start_y, endX=end_x, endY=end_y)#, startX=start_x, startY=start_y, endX=end_x, endY=end_y)