
import cv2  # for web camera
import tensorflow as tf
import os
from scipy.misc import imread
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from waitress import serve
import detect_face  
from triplet_loss import triplet_loss
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os,glob
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
from utils import (
    #load_model,
    get_face,
    get_faces_live,
    #forward_pass,
    save_embedding,
    load_embeddings,
    identify_face,
    allowed_file,
    remove_file_extension,
    save_image
)

app = Flask(__name__)
app.secret_key = os.urandom(24)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
uploads_path = os.path.join(APP_ROOT, 'uploads')
embeddings_path = os.path.join(APP_ROOT, 'embeddings')
allowed_set = set(['png', 'jpg', 'jpeg'])  # allowed image formats for upload

def init():
    # load the pre-trained Keras mode
    sess = tf.Session()
    K.set_session(sess)
    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
    load_weights_from_FaceNet(FRmodel)
    graph = tf.get_default_graph()
    return FRmodel,graph,sess

def img_to_encoding(img1, model):
    img = img1[...,::-1]
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    global graph,FRmodel,sess
    with graph.as_default():
        K.set_session(sess)
        embedding = model.predict_on_batch(x_train)
    return embedding[0]
def who_is_it(image, database, model):
    
    #crop_path=crop_face(image_path)
    encoding = img_to_encoding(image, model)
    
    min_dist = 100
    identity = None
    
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current "emb" from the database.
        dist = np.linalg.norm(db_enc - encoding)

        print('distance for %s is %s' %(name, dist))

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    if min_dist > 0.52:
        return str('None')
    else:
        return str(identity)

@app.route('/upload', methods=['POST', 'GET'])
def get_image():
    """Gets an image file via POST request, feeds the image to the FaceNet model then saves both the original image
     and its resulting embedding from the FaceNet model in their designated folders.

        'uploads' folder: for image files
        'embeddings' folder: for embedding numpy files.
    """

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No 'file' field in POST request!"
            
        list_success=[]        
        #file = request.files['file']
        for file in request.files.getlist('file'):
         
            filename = file.filename
            if filename == "": 
                return "No selected file!"
               

            if file and allowed_file(filename=filename, allowed_set=allowed_set):
                filename = secure_filename(filename=filename)
                try:
                # Read image file as numpy array of RGB dimension
                    img = imread(name=file, mode='RGB')

                    # Detect and crop a 160 x 160 image containing a human face in the image file
                    img = get_face(
                        img=img,
                        pnet=pnet,
                        rnet=rnet,
                        onet=onet,
                        image_size=image_size
                    )

                    # If a human face is detected
                    if img is not None:

                        embedding = img_to_encoding(
                            img,FRmodel
                        )
                        # Save cropped face image to 'uploads/' folder
                        save_image(img=img, filename=filename, uploads_path=uploads_path)

                        # Remove file extension from image filename for numpy file storage being based on image filename
                        filename = remove_file_extension(filename=filename)

                        # Save embedding to 'embeddings/' folder
                        save_embedding(
                            embedding=embedding,
                            filename=filename,
                            embeddings_path=embeddings_path
                        )
                        embedding_dict[filename]=embedding
                        
                        list_success.append(filename)
                        #return "Image uploaded and embedded successfully:- "+str(filename)

                    #else:
                     #   return "Image upload was unsuccessful! No human face was detected!"
                except :
                    return 'error'+str(filename)+'Image uploaded and embedded successfully ' +str(len(list_success))

        return "Image uploaded and embedded successfully:- "+str(len(list_success))

    else:
        return "POST HTTP method required!"
        

@app.route('/predictImage', methods=['POST', 'GET'])
def predict_image():
    """Gets an image file via POST request, feeds the image to the FaceNet model, the resulting embedding is then
    sent to be compared with the embeddings database. The image file is not stored.

    An html page is then rendered showing the prediction result.
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No 'file' field in POST request!"
            

        file = request.files['file']
        filename = file.filename

        if filename == "":
            return "No selected file!"
            

        if file and allowed_file(filename=filename, allowed_set=allowed_set):
            # Read image file as numpy array of RGB dimension
            img = imread(name=file, mode='RGB')

            # Detect and crop a 160 x 160 image containing a human face in the image file
            img = get_face(
                img=img,
                pnet=pnet,
                rnet=rnet,
                onet=onet,
                image_size=image_size 
            )

            # If a human face is detected
            if img is not None:

                
                if embedding_dict:
                    # Compare euclidean distance between this embedding and the embeddings in 'embeddings/'
                    identity = who_is_it(img,embedding_dict,FRmodel
                        
                    )

                    return identity
                    

                else:
                    return "No embedding files detected! Please upload image files for embedding!"
                    

            else:
                return "Operation was unsuccessful! No human face was detected!"
                
    else:
        return "POST HTTP method required!"
        







if __name__ == '__main__':
    embedding_dict = load_embeddings()
    """Server and FaceNet Tensorflow configuration."""
    global graph,FRmodel,sess

    FRmodel,graph,sess=init()
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    image_size = 96
    # Initiate persistent FaceNet model in memory
    facenet_persistent_session = tf.Session(graph=graph, config=config)

    # Create Multi-Task Cascading Convolutional (MTCNN) neural networks for Face Detection
    #pnet, rnet, onet = detect_face.create_mtcnn(sess=facenet_persistent_session, model_path='./npy')
    pnet, rnet, onet = detect_face.create_mtcnn(facenet_persistent_session, './npy')
    # Start flask application on waitress WSGI server
    serve(app=app, host='0.0.0.0')
