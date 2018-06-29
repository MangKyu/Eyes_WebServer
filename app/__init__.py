
from flask import Flask
from app import FaceClassification, DBManager, ObjectMapper

app = Flask(__name__)
fc = FaceClassification.FaceClassification()
db = DBManager.DBConnection()
mapper = ObjectMapper
from app.routes import *