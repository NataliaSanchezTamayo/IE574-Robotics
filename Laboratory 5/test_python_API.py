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
    # make bigger the sensor from the parts producer
        
    #Wait until the objects reach the image sensor then stop!
    # perhaps a loop that asks every x second is the sensor activated or not (read signal)

    time.sleep(10)

    _=vrep.simxStopSimulation(clientID, simx_opmode_blocking)

if __name__ == '__main__':
    main()  