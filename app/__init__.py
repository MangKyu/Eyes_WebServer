from flask import Flask
from app import FaceClassification, DBManager

app = Flask(__name__)
fc = FaceClassification.FaceClassification()
db = DBManager.DBConnection()
from app.routes import *