"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import cv2
import argparse
import tensorflow as tf
import numpy as np
import facenet
import pickle
from align import detect_face
import random
from time import sleep

def main(args):
    modeldir = '../../../../20180402-114759/20180402-114759.pb'
    classifier_filename = '../../../../first_classifier.pkl'
    HumanNames = ["alok@mixorg.com", "arati@mixorg.com", "chetan@mixorg.com", "clarion@mixorg.com", "dhruv@mixorg.com", "mansi@mixorg.com", "mayank@mixorg.com", "mukesh@mixorg.com", "neha@mixorg.com", "parthvee@mixorg.com", "Pradeep_Don", "sachin@mixorg.com", "saketh@mixorg.com", "sanjay@mixorg.com", "shivank@mixorg.com"]
    #sleep(random.random())

    classifier_filename_exp = os.path.expanduser(classifier_filename)
    with open(classifier_filename_exp, 'rb') as infile:
        (model, class_names) = pickle.load(infile)


    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    dataset = facenet.get_dataset(args.input_dir)
    
    print('Creating networks and loading parameters')
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

            facenet.load_model(modeldir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
    
            minsize = 20 # minimum size of face
            threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
            factor = 0.709 # scale factor

            # Add a random key to the filename to allow alignment using multiple processes
            random_key = np.random.randint(0, high=99999)
            bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
            
            
            nrof_images_total = 0
            nrof_successfully_aligned = 0
            

            OUTPUT_RECATEGORIZED = "recategorized"
            for name in HumanNames:
                os.makedirs(os.path.join(os.getcwd(), OUTPUT_RECATEGORIZED,name), exist_ok=True)

            os.makedirs(os.path.join(os.getcwd(),OUTPUT_RECATEGORIZED,"Unknown"), exist_ok=True)

            if args.random_order:
                random.shuffle(dataset)
            for cls in dataset:
                print("cls.name",cls.name)
                output_class_dir = os.path.join(output_dir, cls.name)
                img_count = 0
                if not os.path.exists(output_class_dir):
                    os.makedirs(output_class_dir)
                    if args.random_order:
                        random.shuffle(cls.image_paths)
                for image_path in cls.image_paths:
                    nrof_images_total += 1
                    filename = os.path.splitext(os.path.split(image_path)[1])[0]
                    output_filename = os.path.join(output_class_dir, filename+'.png')
                    
                    #print(image_path)
                    if not os.path.exists(output_filename):
                        try:
                            img = misc.imread(image_path)
                        except (IOError, ValueError, IndexError) as e:
                            errorMessage = '{}: {}'.format(image_path, e)
                            print(errorMessage)
                        else:
                            if img.ndim<2:
                                print('Unable to align "%s"' % image_path)
                                text_file.write('%s\n' % (output_filename))
                                continue
                            if img.ndim == 2:
                                img = facenet.to_rgb(img)
                            img = img[:,:,0:3]

                            bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                            nrof_faces = bounding_boxes.shape[0]
                            if nrof_faces>0:
                                det = bounding_boxes[:,0:4]
                                det_arr = []
                                img_size = np.asarray(img.shape)[0:2]
                                if nrof_faces>1:
                                    if args.detect_multiple_faces:
                                        for i in range(nrof_faces):
                                            det_arr.append(np.squeeze(det[i]))
                                    else:
                                        bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                                        img_center = img_size / 2
                                        offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                                        offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                                        index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                                        det_arr.append(det[index,:])
                                else:
                                    det_arr.append(np.squeeze(det))

                                FOR_STORAGE = []
                                FOR_FEED = []

                                for i, det in enumerate(det_arr):
                                    det = np.squeeze(det)
                                    bb = np.zeros(4, dtype=np.int32)
                                    bb[0] = np.maximum(det[0]-args.margin/2, 0)
                                    bb[1] = np.maximum(det[1]-args.margin/2, 0)
                                    bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
                                    bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
                                    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                                    scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
                                    FOR_STORAGE.append(scaled)
                                    prewhitened = facenet.prewhiten(cv2.resize(scaled, (160,160),
                                                    interpolation=cv2.INTER_CUBIC))
                                    FOR_FEED.append(prewhitened.reshape(-1,160,160,3))
                                    nrof_successfully_aligned += 1
                                    #output_filename_n = "{}_{}.{}".format(output_filename.split('.')[0], i, output_filename.split('.')[-1])
                                    #misc.imsave(output_filename_n, scaled)
                                    #text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
                                
                                Final_reshaped = np.array(FOR_FEED).reshape(-1,160,160,3)
                                emb_array = np.zeros((Final_reshaped.shape[0], embedding_size))
                                feed_dict = {images_placeholder: Final_reshaped, phase_train_placeholder: False}
                                emb_array[:, :] = sess.run(embeddings, feed_dict=feed_dict)
                                predictions = model.predict_proba(emb_array)
                                best_class_indices = np.argmax(predictions, axis=1)
                                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                                for i, best_class_index in enumerate(best_class_indices):
                                    file_name = HumanNames[best_class_index]+".png"
                                    if best_class_probabilities[i] > 0.63:
                                        file_name = HumanNames[best_class_index]+".png"
                                        output_filename_n = "{}_{}_{}.{}".format(file_name.split('.')[0], img_count,i, file_name.split('.')[-1])
                                        misc.imsave(os.path.join(os.getcwd(),OUTPUT_RECATEGORIZED,HumanNames[best_class_index],output_filename_n), FOR_STORAGE[i])
                                    else:
                                        file_name = "Unknown.png"
                                        output_filename_n = "{}_{}_{}.{}".format(file_name.split('.')[0], img_count, i, file_name.split('.')[-1])
                                        misc.imsave(os.path.join(os.getcwd(),OUTPUT_RECATEGORIZED,"Unknown",output_filename_n), FOR_STORAGE[i])
                    
                            else:
                                print('Unable to align "%s"' % image_path)
                    img_count += 1
            
            
       


      


    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
            

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order', 
        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.6)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=True)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
