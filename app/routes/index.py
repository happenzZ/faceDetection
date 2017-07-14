# -*- coding: UTF-8 -*-
from app import app
import json

from flask import request

from app import faceDetection

@app.route('/')
def index():
    return json.dumps('Hello world!')

#
# @app.route('/cropOneImage')
# def _cropOneImage():
#     return json.dumps('test')
#
#
# @app.route('/cropOneFolder')
# def _cropOneFolder():
#     return json.dumps('test')

#
# @app.route('/api/update/video/<videoId>', methods=['GET'])
# def processVideo(videoId):
#     faceDetection.process_video(videoId)
#     return 'video {} processing finished'.format(videoId)

if __name__ == '__main__':
    pass
