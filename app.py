import numpy as np
import cv2
from keras.model import load_model
from flask import Flask, render_template, Response
import tensorflow as tf
global graph
global writer
from skimage.transform import resize

graph = tf.get_default_graph()
writer = None 

model = load_model("/content/drive/MyDrive/Colab Notebooks/iceberg_model.h5")

app = Flask(__name__)

print("[INFO] accessing video stream")
vs = cv2.VideoCapture("")

pred = ""

def detect(frame):
    img = resize(frame,(75,75))
    img = np.expand_dims(img,axis=0)
    if(np.max(img)>1):
        img = img/255.0
    with graph.as_default():
        prediction = model.predict_classes(img)
    pred=prediction[0][0]
    if not pred:
        text = "Beware!! Iceberg ahead. "
    else:
        text = "You are safe! Its a Ship"
    return text,pred 

def gen():
    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break 
        data,pred = detect(frame)
        
        text = data 
        cv2.putText(frame, text, (10, frame.shape[0] - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
        cv2.imwrite("",frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key==ord("q"):
            break 
        fourcc = cv2.VideoCapture(*"MJPG")
        writer = cv2.VideoWriter()            
                    
        
