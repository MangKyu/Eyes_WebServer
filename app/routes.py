import datetime
from flask import render_template, request, jsonify
from app import app, db, mapper
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


@app.route("/getPatient", methods=['POST'])
def get_patient():
    user_id = request.values.get('userId')
    patient_id = db.get_patient_id(user_id)
    patient_info = db.get_patient_info(patient_id)
    patient_vo = mapper.get_patient(patient_info)
    return jsonify(patient_vo.serialize())


@app.route("/getHistories", methods=['POST'])
def get_patient_info():
    patient_id = request.values.get('patientId')
    patient_histories = db.get_histories(patient_id)
    history_list = mapper.get_histories(patient_histories)
    return jsonify(history_list)


@app.route("/getPath", methods=['GET'])
def get_path():
    startX = request.args.get('startX')
    startY = request.args.get('startY')
    endX = request.args.get('endX')
    endY = request.args.get('endY')

    return render_template('path.html', startX=startX, startY=startY, endX=endX, endY=endY)


@app.route("/path", methods=['GET'])
def path():
    start_x = request.args.get('startX')
    start_y = request.args.get('startY')
    return render_template('map.html', start_x=start_x, start_y=start_y)