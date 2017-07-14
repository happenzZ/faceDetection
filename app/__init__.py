# -*- coding: utf-8 -*-

from flask import Flask
from flask_cors import CORS

from app.faceDetection.faceDetection import FaceDetection

app = Flask(__name__)
app.config['DEBUG'] = True
app.config['TEMPLATES_AUTO_RELOAD'] = True

# flask_cors: Cross Origin Resource Sharing (CORS), making cross-origin AJAX possible.
CORS(app)

# define faceService
faceDetection = FaceDetection()

from app.routes import index