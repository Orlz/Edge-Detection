#!/usr/bin/env python

"""
==================
EAST Text Detector
==================

EAST (An Efficient and Accurate Scene Text detector)
https://arxiv.org/abs/1704.03155v2 (LOOK AT HOW TO REFERENCE PROPERLY)

Some important elements to note from this script: 

    net (line 63)  = This is the main brain of the text detector system. 
                     It is essentially a neural network developed by the EAST team which was trained on a vast array of images, 
                     to be able to detect all kinds of text from such images. This ranged from pictures of the names and numbers
                     on football players jersey's taken from action pictures of the match to the text on buses and much more. 
                     
                     
    text_detector  = This function preprocesses the images to get them into the format to work with East. It then then runs 
                     it through the EAST net and draws a rectangular ROI around the areas it believes to be text. 
                     
This script has been inspired by Adrian Rosebrock's use of EAST for text detection in videos. This can be found at the following address: https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/ 

Usage: 
$ python3 src/ROI_detector.py 
                     
"""

"""
============
Dependencies
============
    - numpy:                      To work with numerical arrays of the images 
    - cv2 :                       To deal with the image processing functionality 
    - non_max_suppression:        A function called from the imutils script stored in our utils folder
    - time:                       Need to figure this out
"""

#Functions from the utils folder
#import utils.imutils as functions
import os
import sys
sys.path.append(os.path.join(".."))
import numpy as np
import time
import cv2
import argparse
from utils.imutils import non_max_suppression



"""
==================
Running the Script
==================
"""

#Setup 
print ("Hey there, let's try to detect where the text is in this graffiti!") 


#Calling the EAST library
print("\n I'm just calling the EAST network")
net = cv2.dnn.readNet("frozen_east_text_detection.pb")


#Defining a function to detect text from an image 
def text_detector(image):
    orig = image                                     # Name the original image
    (Height, Width) = image.shape[:2]                # Get the height and width using the shape function in cv2

    """
    Step One: Resize the image
              The East detector requires images to be in a shape size where both height and text are multiples of 32.
              This is just because of the way the neural-network was trained, on images of standardised sizes.
    """
    
    (newW, newH) = (640, 320)                        # Creating values to resize the image to fit with the EAST package 
    rW = Width / float(newW)                         # Ensuring image width will be a multiple of 32 (requirement) 
    rH = Height / float(newH)                        # Ensuring image height will be a multiple of 32 (requirement)

    image = cv2.resize(image, (newW, newH))          # Then we'll resize the image into a 640 by 320 rectangle
    (Height, Width) = image.shape[:2]

    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",             # An output layer of EAST which informs on the confidence of identifying text 
        "feature_fusion/concat_3"]                   # Tells us about the orientation of the text 

    
    """
    Step 2: Preprocessing: Known as "Block from Image"
    - This is part of the neural network library in OpenCV
            Essentially, this is a way of standardising the images. It works by finding the mean of the image for all 3 red, blue
            and green colour channels and subtracts these to take away the influence of brightness, light exposure etc. 
            This makes the neural network more robust and generalizable when new images are fed in. 
    """

    blob = cv2.dnn.blobFromImage(image, 1.0, (Width, Height),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)     # Values are the mean RGB values calculated from initial images
                                                               # This allows us to use the same RGB values as EAST does

        
    net.setInput(blob)                                         #Employs the subtraction & standarfization of the image
    (scores, geometry) = net.forward(layerNames)               #Gives the scores and geometries from layers (line X and X)

    (numRows, numCols) = scores.shape[2:4]                     #Defines number of columns & rows of the image shape
    rects = []                                                 #List to store bounding box (bb) coordinates of text 
    confidences = []                                           #List to store the probability associated with each bb

    for y in range(0, numRows):                                # This loop helps to identify where the text is in the image 
                                                               # The loop extracts the scores and geometry data for row y 
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns of this current row (y) 
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            # This filters out weak text detectionss
            if scoresData[x] < 0.5:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then 
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # drawing the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # Using the corrdinates to make and draw the bounding box
            # And adding the probability score to the respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    #This function ensures we're using the most confident box and not drawing hundreds around the same chunk of text 
    #It works by suppressing all the lower value scores and taking only the most confident ones
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    
    """
    Step 3: Draw rectangular ROI on the original image (not the processed) 
    """
    for (startX, startY, endX, endY) in boxes:

        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # draw the bounding box on the image
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 3)
    return orig
   
"""
==========================================
Create loop to run through image directory 
==========================================
"""

print("\nWe're all set up so I'm going to loop through your images and see if I can detect any text among them") 

"""
List the images 
"""
#As there's just a few images we'll list them here 

image1 = 'Berlin'
image2 = 'Bosnia'
image3 = 'N_Ireland'
image4 = 'Syria'
image5 = 'US'
image6 = 'weholdtruths'


#Create a list of the images in the directory 
image_array = [image1, image2, image3, image4, image5, image6]

"""
Create a loop which runs through all images in the list and runs EAST text detection on them 
"""

#For every image in the image_array list
for image in image_array:
    #Get the image name
    image_name = image
    #Get the path to the image 
    image_path = "Images/"+image_name+".jpeg"
    #Read in the image 
    image = cv2.imread(image_path)
    #Resize the image to fit with the EAST dimensions 
    imageO = cv2.resize(image, (640,320), interpolation = cv2.INTER_AREA)
    #Create a copy of the original image 
    imageX = imageO
    #run the text_detector function defined above on the image 
    orig = text_detector(imageO)
    #Save the image in the output folder
    cv2.imwrite("Output/"+image_name+"_EAST.jpeg", orig)
    
#Close off the script 
print("Good news - it looks like we've found some. You can check out what I have detected in your Output folder!")
print("\n The images are saved with their LocationName_EAST.jpeg") 

