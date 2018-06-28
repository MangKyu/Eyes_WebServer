from flask import Flask
from app import FaceClassification

app = Flask(__name__)
fc = FaceClassification.FaceClassification()

from app.routes import *