# -*- coding: UTF-8 -*-
import os

import cv2
import numpy as np
import tensorflow as tf
from scipy import misc

from app.faceDetection.align import detect_face


class FaceDetection(object):
    def __init__(self, gpu_memory_fraction=0):
        # parameters
        self.gpu_memory_fraction = gpu_memory_fraction

        # graph and session:
        self.detection_graph = None
        self.detection_sess = None

        # mtcnn
        self.pnet = None
        self.rnet = None
        self.onet = None

        self.init_config()
        return

    def __del__(self):
        """
            frees all resources associated with the session
        """
        self.detection_sess.close()
        return

    def init_config(self):
        """
            initial configuration
        :return:
        """
        self.load_detection_model()
        return

    ################################
    # Face faceDetection
    ################################
    def load_detection_model(self):
        """load MTCNN model"""
        print('Loading MTCNN model')
        self.detection_graph = tf.Graph()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
        self.detection_sess = tf.Session(graph=self.detection_graph,
                                         config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with self.detection_graph.as_default():
            print('detection_sess: ', self.detection_sess)
            with self.detection_sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.detection_sess,
                                                                           os.path.dirname(__file__) + '/align/')
        return

    def detect_face(self, img, prob_threshold):
        """
            detect face bounding boxes from each frame
        :param img: one frame image (RGB)
        :param prob_threshold: probability threshold for face faceDetection
        :return bounding_boxes: np.array([[left, top, right, bottom, probability], [...], ...])
        """
        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        bounding_boxes, _ = detect_face.detect_face(img, minsize, self.pnet, self.rnet, self.onet, threshold,
                                                    factor)
        bounding_boxes = np.array([item for item in filter(lambda x: x[4] > prob_threshold, bounding_boxes)])
        return bounding_boxes

    def prewhiten(self, x):
        """
            token from facenet, prewhiten an image
        :param x: an input image
        :return: a prewhiten image
        """
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        return y

    def crop_face(self, img, bounding_boxes, image_size, margin):
        """
            crop face from each frame based on bounding boxed detected
        :param img: one frame image (RGB)
        :param bounding_boxes: bounding boxes result from face faceDetection, [[left, top, right, bottom, prob], [...], ...]
        :param image_size: resize image, 160*160
        :param margin: Margin for the crop around the bounding box (height, width) in pixels, 44
        :return: crop_images(RGB, 160*160), prewhitened_images(160*160), grayed_images(GRAY, 48*48)
        """
        img_size = np.asarray(img.shape)[0:2]
        nrof_faces = bounding_boxes.shape[0]
        crop_images = []
        prewhitened_images = []
        grayed_images = []
        if nrof_faces > 0:
            for faceIdx in range(nrof_faces):
                det = np.squeeze(bounding_boxes[faceIdx, 0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                grayed = cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY)
                # resize grayed for face expression
                grayed = cv2.resize(grayed, (48, 48))
                prewhitened = self.prewhiten(aligned)
                crop_images.append(aligned)
                prewhitened_images.append(prewhitened)
                grayed_images.append(grayed)
            grayed_images = np.array(grayed_images, dtype=np.uint8).reshape((-1, 48, 48, 1))
        return crop_images, prewhitened_images, grayed_images

    def detect_one_face(self, img, prob_threshold):
        """
            detect face bounding boxes from each frame
        :param img: one frame image (RGB)
        :param prob_threshold: probability threshold for face faceDetection
        :return bounding_boxes: np.array([[left, top, right, bottom, probability], [...], ...])
        """
        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        bounding_boxes, _ = detect_face.detect_face(img, minsize, self.pnet, self.rnet, self.onet, threshold,
                                                    factor)
        bounding_boxes = bounding_boxes.tolist()
        if len(bounding_boxes) < 1:
            return None
        # prob max => [left, top, right, bottom]
        bounding_box = sorted(bounding_boxes, key=lambda k: k[-1], reverse=True)[0]
        return bounding_box

    def crop_one_face(self, img, bounding_box, image_size, margin):
        if not bounding_box:
            return None
        img_size = np.asarray(img.shape)[0:2]
        det = np.squeeze(bounding_box[0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        return aligned

    def save_one_face(self, img, img_path):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(img_path, img)
        return

    def cut_face_for_image(self, input_path, output_path):
        img = cv2.imread(input_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bounding_box = self.detect_one_face(img, 0.95)
        if not bounding_box:
            print('no face in this image!')
            return None
        crop_image = self.crop_one_face(img, bounding_box, 160, 44)
        self.save_one_face(crop_image, output_path)
        return output_path

    def cut_face_for_folder(self, input_folder, output_folder):
        input_folder = os.path.abspath(input_folder)
        output_folder = os.path.abspath(output_folder)
        output_path_list = []
        img_list = [x for x in os.listdir(input_folder) if x.endswith('.JPG')]
        for img in img_list:
            img_name = img.split('.')[0]
            img_input_path = os.path.join(input_folder, img)
            img_output_path = os.path.join(output_folder, '{}.jpg'.format(img_name))
            output_path_list.append(self.cut_face_for_image(img_input_path, img_output_path))
        return output_path_list

if __name__ == '__main__':
    faceDetection = FaceDetection()
    # image = cv2.imread('../data/anami_kids/IMG_0000.JPG')
    # print('image.shape: ', image.shape)
    #
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # bounding_boxes = faceDetection.detect_face(image, 0.95)
    # print('bounding_boxes: ', bounding_boxes)
    # print('bounding_boxes.shape: ', bounding_boxes.shape)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # input_path = '../data/anami_kids/IMG_0000.JPG'
    # output_path = '../data/result/IMG_0000.jpg'
    # faceDetection.cut_face_for_image(input_path, output_path)

    input_folder = '../data/anami_kids'
    output_folder = '../data/result'
    faceDetection.cut_face_for_folder(input_folder, output_folder)
