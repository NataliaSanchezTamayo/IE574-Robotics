# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:58:15 2020

@author: Natalia Sanchez-Tamayo

"""
import vrep 
import sys
import time
import numpy as np
from vrepConst import *
from PIL import Image
from tensorflow import keras
from tensorflow.keras import metrics
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D,MaxPooling2D,Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical


def get_image_from_vrep(clientID,cam_handle):
    iter=60
    for i in range(iter): # repeat iter number of times
        time.sleep(1)
        # Check if there is an object under the camera:
        _,signal=vrep.simxGetIntegerSignal(clientID,"takeImage",vrep.simx_opmode_blocking)
        
        if int(signal)==1:
            print("got image")
            #There is an object under the camera
            err, resolution, image = vrep.simxGetVisionSensorImage(clientID, cam_handle, 0, vrep.simx_opmode_blocking)
            img = np.array(image,dtype=np.uint8)
            print(img.shape,"shape")
            img.resize([resolution[1],resolution[0],3])
            
            if err == vrep.simx_return_ok:
                im = Image.fromarray(img)
                im.save("current_image.png")
    
            elif err == vrep.simx_return_novalue_flag:
                print ("no image yet")
                pass
            else:
              print (err)
            return image
          

def predict_class(classifier,img_width, img_height):
    img_str="current_image.png"
    # PREDICT THE CLASS OF ONE IMAGE
    img = image.load_img(img_str, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = classifier.predict_classes(images, batch_size=1)
    classes_predic = classifier.predict(images, batch_size=1)
    class_names=("carton","glass","metal","plastic")
    print ("predicted the class",class_names[classes[0]],"code:", classes[0],)
    return classes[0],class_names[classes[0]]

def load_saved_model(model_name):
    classifier=load_model(model_name)  
    print(classifier.summary())
    return classifier

def pick_and_classify(clientID,class_code):
    print("Sent command")
    inputInts=[class_code]
    inputFloats=[0.0]
    inputStrings=["class_name"]
    inputBuffer=bytearray()
    # inputBuffer.append(78)
    res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(clientID,'IRB140',vrep.sim_scripttype_childscript,
               'pick_and_classify',inputInts,inputFloats,inputStrings,inputBuffer,simx_opmode_blocking)
    time.sleep(3)
    print("finished sending pick up command")
    return(res)


def main():
    # this is the main script where all the code is executed
    # access all the VREP elements
    vrep.simxFinish(-1) 
    # just in case, close all opened connections
    clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5) 
    # start aconnection
    if clientID!=-1:
        print ("Connected to remote API server")
    else:
        print("Not connected to remote API server")
        sys.exit("Could not connect")

    _=vrep.simxStartSimulation(clientID,vrep.simx_opmode_blocking)

    res, v1 = vrep.simxGetObjectHandle(clientID, 'blobDetectionCamera_camera', vrep.simx_opmode_blocking)

    img_width=64
    img_height=64
    # load the classifier model saved in classifier.h5
    # classifier=load_saved_model("classifier.h5") 
    classifier=load_saved_model("pretrained_classifier.h5") 
    num_repeat=5 # the number of times out classifier will run

    for i in range(num_repeat):
        vrep_image=get_image_from_vrep(clientID,v1)
        class_code,class_name=predict_class(classifier,img_width, img_height)
        ret=pick_and_classify(clientID,class_code)

    # wait 50 seconds before stopic the simulation
    time.sleep(50)

    _=vrep.simxStopSimulation(clientID, simx_opmode_blocking)

if __name__ == '__main__':
    main()  