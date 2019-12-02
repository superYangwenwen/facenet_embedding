from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import cv2
from PIL import Image
import numpy as np
import facenet
from align import detect_face
import os
import time
import pickle
import math
from statistics import mode, StatisticsError
from pathlib import Path


from datetime import datetime


modeldir = './20180402-114759/20180402-114759.pb'

sess = None


def get_embeddings(image_array):

    if len(image_array) < 1:
        return None
    for impath in image_array:
        my_file = Path(impath)
        if not my_file.is_file():
            print("File not found: "+ impath)
            return None
    
    

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():

            ####    LOADING OF FACE DETECTION MODEL     ####
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

            minsize = 40  # minimum size of face
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor
            input_image_size = 160
            
            
            InstStartTime = datetime.now().strftime('%Y%m%d-%H_%M_')
            print('Loading Modal')

            ####    EMBEDDING ML MODEL LOADING AND TENSOR SETTING STUFF   ####

            facenet.load_model(modeldir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            c = 0
            curTime = time.time()+1    # calc fps

            scaled_reshape = []

            ####    READING EACH IMAGE AND GENERATING   ####

            for impath in image_array:
                frame = cv2.imread(impath)

                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)

                frame = frame[:, :, 0:3]
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]        #NUMBER OF FACES IN A PHOTO
                # IT SHOULD BE 1 in each photo
           
                if nrof_faces == 1:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]
                    cropped = []
                    scaled = []
                    savable = []
                    bb = np.zeros((1,4), dtype=np.int32)
                        

                    bb[0][0] = det[0][0]
                    bb[0][1] = det[0][1]
                    bb[0][2] = det[0][2]
                    bb[0][3] = det[0][3]

                    # inner exception
                    if bb[0][0] <= 0 or bb[0][1] <= 0 or bb[0][2] >= len(frame[0]) or bb[0][3] >= len(frame):
                        #print('Face is near EDGE!')
                        return

                    cropped.append(frame[bb[0][1]:bb[0][3], bb[0][0]:bb[0][2], :])
                    scaled.append(np.array(Image.fromarray(cropped[-1]).resize((input_image_size, input_image_size), resample=Image.BICUBIC)))
                    scaled[-1] = facenet.prewhiten(scaled[-1])
                    scaled_reshape.append(scaled[-1].reshape(-1,input_image_size,input_image_size,3))


            if len(scaled_reshape) >0:
                print("LENGTH OF ARRAY", len(scaled_reshape))
                array_for_feed = np.array(scaled_reshape).reshape(-1,input_image_size,input_image_size,3)
                emb_array = np.zeros((array_for_feed.shape[0], embedding_size))
                feed_dict = {images_placeholder: array_for_feed, phase_train_placeholder: False}
                emb_array[:, :] = sess.run(embeddings, feed_dict=feed_dict)             ## THIS IS THE LINE THAT GENERATES EMBEDDINGS

                
                return emb_array
            return None


####    calling the function    ####

images = ["rawImage.jpg","top-model.jpg"]
embedds = get_embeddings(images)
print (embedds.shape)

