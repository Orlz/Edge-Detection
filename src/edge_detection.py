#!/usr/bin/env python

"""
==============================
Assignment 01: Edge Detection 
==============================

This assignment tests our ability to find and define edges in an image with many engraved letters. The script completes 4 tasks: 

For a given image
  1. Find the ROI on the image and draw a rectangle around it   (save as image_with_ROI.jpg)  
  2. Using this ROI, crop the image to only include the ROI   (save as image_cropped.jpg) 
  3. Use canny edge detection techniques to identify every letter in the image image_letters.jpg) 
  4. Draw green contours around each letter  (save image as  
  
We are using argparse and have 3 input parameters which need defined in the command line:
    image_path:    str <path-to-image>
    roi:           x1 y1 x2 y2  <the coordinate points defining the top-left and bottom-right corner of ROI (to draw a rectangle)> 
    output_path:   str <path-to-output-file>
    
Example of command line input:
    src/edge_detection.py --image_path <path-to-image> --roi x1 y1 x2 y2 --output_path <path-to-output-file>

Worked example of how command line code would look: 
$ python src/edge_detection.py --image_path ../Images/weholdtruths.jpeg --roi 1400 890 2900 2800 --output_path output2/
      
"""


"""
Import the Dependencies 
"""

#operating systems 
import os
import sys
sys.path.append(os.path.join(".."))

#Image processing 
import cv2
import numpy as np

#Command line functionality
import argparse


"""
Main Function with Argparse arguments
""" 
def main():
    """
    Setting up our arguments with argparse
    """
    
    # initialise argparse 
    ap = argparse.ArgumentParser()

    #Argument 1: Path to the images 
    ap.add_argument("-i", "--image_path", 
                    required = False, 
                    default = ("Images/weholdtruths.jpeg"), 
                    help = "Path to image")
    
    #Argument 2: Define the region of interest points 
    ap.add_argument("-r", "--roi", 
                    required = True, 
                    help = "Points of ROI in image", 
                    nargs='+')   #nargs helps control no. inputs
    
    #Argument 3: Define the output path 
    ap.add_argument("-o", "--output_path", 
                    required = True, 
                    help = "Path to output directory")
    
    # parse arguments
    args = vars(ap.parse_args())
    
    
    """
    Assigning our arguments to variable names for the script
    """
    image_path = args["image_path"]            # Connect the image_path variable to the defined image path from command line
    ROI = args["roi"]                          # Connect the ROI variable to the 4 defined numbers in the command line      
    
    
    """
    Create the output directory
    """
    output_path = args["output_path"]          #Connecting the output_path variable to the command lin output path 
    if not os.path.exists(output_path):        #If this output directory does not exist, 
        os.mkdir(output_path)                  #Then create one with the name defined in the command line
        
        
    
    """
    Operationalising the script 
    """ 
    #Let the user know the script is about to begin 
    print("Hey there, I'm about to conduct some edge detection on your chosen image!")
 
    #Read in the image with cv2 
    image = cv2.imread(image_path)
    
    #Create a name to save this image with later
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    

    """
    Task 1: Draw the Region of Interest (RIO)  
    """ 

    print(f"\nFirst, I'm drawing a red rectangle around your ROI. This will be saved as {image_name}_with_ROI.jpg")
    
    # First, define top left corner using first 2 roi inputs (x1, y1), and bottom right corner using last 2 roi inputs (x2, y2) 
    top_left_corner = (int(ROI[0]), int(ROI[1]))   
    bottom_right_corner = (int(ROI[2]), int(ROI[3]))
    
    # Next, draw a red rectangle as the ROI onto a copy of the image
    image_ROI = cv2.rectangle(image.copy(), top_left_corner, bottom_right_corner, (0,0,255), (2))
    
    # Finally, save the image with ROI (using the image_name) 
    cv2.imwrite(os.path.join(output_path, f"{image_name}_with_ROI.jpg"), image_ROI)  
    
    """
    Task 2: Crop the Image   
    """           
    
    print(f"\nThat's complete. Now I'm cropping your image and will save it as {image_name}_cropped.jpg. ")
    
    # Use the ROI points to crop the image (this can be read as cropped image = image[startY:endY, startX:endX])
    image_cropped =image[top_left_corner[1]:bottom_right_corner[1], top_left_corner[0]:bottom_right_corner[0]] 
    
    # save cropped image to the output path 
    cv2.imwrite(os.path.join(output_path, f"{image_name}_cropped.jpg"), image_cropped)

                
    """
    Task 3: Conduct the image processing with canny edge detection    
    """ 
    
    print("\nThat's done so we'll now apply the canny edge detection")

    # Blur the image to remove high freq.edges (we'll use gaussian blurring with a (7x7) kernal and 0 for the default sigma) 
    image_blurred = cv2.GaussianBlur(image_cropped, (7,7), 0)
    
    # We then apply thresholding to make the image black & white (this helps to segment foreground and background)
    # This can be read as (cv2.threshold(image, threshold_value, colour, method )) threshold = 110, colour = white (255)
    (_, image_binary) = cv2.threshold(image_blurred, 110, 255, cv2.THRESH_BINARY)
    
    # apply cv2's canny edge detection (We've manually set the parameters to 60 and 150) 
    image_canny = cv2.Canny(image_binary, 60, 150)
 
                
    """
    Task 4: Draw the contours    
    """
    
    print(f"\nIt's looking good. I'm just drawing the contours now - this image will be saved as {image_name}_letters.jpg")
                
    # First we find the contours with cv2's findContours function 
    (contours, _) = cv2.findContours(image_canny.copy(),               #Use a copy of the canny_image 
                                     cv2.RETR_EXTERNAL,                #Set the contour retrieval mode to External
                                     cv2.CHAIN_APPROX_SIMPLE)          #Use the chain_approx_simple approximation method
    
                
    # Then, we draw the contours on the cropped image with thickness of 2 (using a copy of the cropped image) 
    image_letters = cv2.drawContours(image_cropped.copy(), contours, -1, (0,255,0), 2)
    
    # Saving the final image 
    cv2.imwrite(os.path.join(output_path, f"{image_name}_letters.jpg"), image_letters)
                 
                
    """
    Let the user know the scipt is complete 
    """
                
    print(f"\nThat's you finished - woohoo! The images are saved in {output_path}.")
                
    
# complete the main function script    
if __name__=="__main__":
    main()
