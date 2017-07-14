# -*- coding: UTF-8 -*-
from app import app
import json

from flask import request

from app import faceDetection

@app.route('/')
def index():
    return json.dumps('Hello world!')

#
@app.route('/cropOneImage', methods=['POST'])
def _cropOneImage():
    post_data = json.loads(request.data.decode())
    input_path = post_data['inputPath']
    output_path = post_data['outputPath']
    result = faceDetection.cut_face_for_image(input_path, output_path)
    return json.dumps(result)

@app.route('/cropOneFolder', methods=['POST'])
def _cropOneFolder():
    post_data = json.loads(request.data.decode())
    input_folder = post_data['inputFolder']
    output_folder = post_data['outputFolder']
    result = faceDetection.cut_face_for_folder(input_folder, output_folder)
    return json.dumps(result)

#
# @app.route('/api/update/video/<videoId>', methods=['GET'])
# def processVideo(videoId):
#     faceDetection.process_video(videoId)
#     return 'video {} processing finished'.format(videoId)

if __name__ == '__main__':
    pass
