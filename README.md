[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![describtor - e.g. python version](https://img.shields.io/badge/Python%20Version->=3.6-blue)](www.desired_reference.com)  ![](https://img.shields.io/badge/Software%20Mac->=10.14-pink)


# Edge Detection

**Visual Analytics: Portfolio Assignment 01**




<div align="center"><img src="https://github.com/Orlz/CDS_Visual_Analytics/blob/main/Portfolio/artificial-intelligence.png" width="100" height="100"/></div>






Edge detection is an important element of computer vision and often seen as the beginning step towards tearching computers how to read and interpret images. It is a process whereby convolutional kernels are used to find the greatest rate of change among pixel intensities in an image (i.e. where in the image does the pixel intensity quickly change from a low level of intensity (indicating a dark colour) to a high intensity (indicating a light colour). Where there is a notable change, it is reasonable to assume that this will be an edge. 

Convolutional kernels are used to help to manage the discrete nature of pixel intensities, allowing us to work with the image as through the intensity values were on a continuous scale rather than an integer between 0 to 255. In this assignment, we employ canny edge detection due to its better generalizability over the sobel and laplacian kernels. A key element of canny edge detection is that it works using hysterisis thresholding, which means only the maximums and the pixels connected to these are considered part of the edge. Any pixels which do not pass the threshold are not considered. This helps the kernal to manage noise and generally results in a clearer edge being identified. 

## Table of Contents 

- [Assignment Description](#Assignment)
- [Scripts and Data](#Scripts)
- [Methods](#Methods)
- [Operating the Scripts](#Operating)
- [Discussion of results](#Discussion)


## Assignment Description

In this assignment, we attempt to extract specific features from images using canny edge detection and contours.In particular, we're going to see if we can find text in an image with many engraved letters. The script completes 4 basic tasks: 

For a given image
  1. Find the region of interest (ROI) on the image and draw a rectangle around it     (save as image_with_ROI.jpg)  
  2. Using this ROI, crop the image to only include the ROI                            (save as image_cropped.jpg) 
  3. Use canny edge detection techniques to identify every letter in the image          
  4. Draw green contours around each letter of the letters detected                    (save as image_letters.jpg) 
  
### Assignment Expansion: Detect images in graffiti collected from areas of political tension 

While it's great to be able to manually find the ROI of images and use this to find the contours of text on one image, the task quickly becomes cumbersome if we have a collection of text images which need processed together. Say for example, we're looking at cultural graffiti from around the world and we want to be able to detect the messages hidden among paint-stricken walls. How well would our method generalise here?

An additional script has therefore been added to the assignment as part of an exploratory extension, looking at how we could make a more generalizable script to find the region's of interest for images of cultural graffiti. 5 images have been seleted from areas commonly known for their political graffiti and streetart. These can be found in the Images folder: 

___The Image collection: Cultural Graffiti___

| Script | Description|
|--------|:-----------|
Berlin | This image is taken from the Eastside Gallery in Belin. It captures text written on the Berlin Wall not long after it fell reading: "A few more walls like this should fall" 
Bosnia | This image is taken from Sarajevo, the captial of Bosnia. The image displays a soldier claiming to defend the city.
Northern Ireland | This image was taken after the peace treaty in Northern Ireland and displays text on a gate which reads "There never was a good war or a bad peace"
United States | This image is taken from the Black Lives Matter movement in the US which flared up in 2020.
Syria | This image is taken from a collapsed building in Syria, where a local street artist leaves powerful messages, surprisingly in English.


The script employs EAST text detection to try to find text within the image and draw a region of interest around it. It takes the manual steps out of the process and moves us towards considering how we could make our original script more robust. As we'll be diving deep in other areas of the portfolio, the contour detection has not been run on these EAST processed images, but could easily be done by creating a loop through the images to conduct the cropping, finding contours, and drawing these contours on the images. This process has been demonstrated on a single image in a Notebook file found in the src folder. 


## Scripts and Data


There are 2 scripts included in this assignment which are located in the 'src' folder. Full details of how each script works can be found in the script itself.

```
Script             | Description
----------------   | ------------
edge_detection.py  | Part 1: This is the original script for conducting edge detection on the Jefferson monument
ROI_detector.py    | Part 2: This script is the exploratory extension, using EAST text detection 
```



The image data for the project can be found in the 'Images' folder. 

```
        | Data
------- | ---------
Part 1  | Part 1: This uses the one image called 'weholdtruths.jpeg'. The image is of the Jefferson memorial.  
Part 2  | Part 2: This uses the images in the cultural graffiti collection as well as the Jefferson memorial. 
```

The original image can be downloaded by the user [here](https://upload.wikimedia.org/wikipedia/commons/f/f4/%22We_Hold_These_Truths%22_at_Jefferson_Memorial_IMG_4729.JPG)






## Methods 

### Part 1

The Canny edge detection was conducted using a number of the image processing functions in the OpenCV package. OpenCV is an open-source computer vision and machine learning software library with a number of algorithms defined for use in computer vision. The first two steps were completed by reading the image using imread() and determining the dimensions for the region of interest using the shape() function. This proceedure was essentially estimation and playing around with the ROI dimensions until the retangle fit tightly around the text. These ROI dimensions were used to crop the image, thereby completing task 1 and 2. 

The image was then passed on to the preprocessing steps for canny edge detection. These included blurring the image (to remove high frequency edges) and converting it to greyscale using a binary thresholding method. This binary method converts the pixels to either black or white and is used to help segment the foreground and background. The threshold was set to 110, meaning that pixels of intensity 110 or above would be considered white and below would be considered black. Cv2’s canny edge detection function was then applied, which computes the sobel gradients in both the X (horizontal) and Y (vertical) directions to detect the edges. 

Finally, the findContours() function was used to locate and join any curves with consistent pixel intensity, using chain_approximation_simple as the approximation parameter. 

### Part 2

The EAST text detection was completed using a script inspired by Adrian Rosebrock from PyImageSearch. It employs a function which first processes the images to resize their dimensions into a multiple of 32 (to fit within the specifications of the model), standardises them by taking away their brightness and exposure (to make them more recognisable for the model), and feeds them into the model which identifies areas where it suspects text to be using their coordinate location. The model applies some non-max-suppression using a confidence interval to only identify the most probably occurrences of text, as opposed to drawing many ROI’s on the image. The rectangle ROI box is then drawn in much the same way as in part 1, by identifying the top left and bottom right corner. A loop has been constructed to loop through the 6 images and process them in turn, to avoid having to run the script multiple times. 

This function is rather complex, but each step is well documented and explained in the script. The reader is encouraged to look there for more information on the individual processing steps taken (ROI_detector.py). 



## Operating the Scripts 

There are 3 steps to take to get your script up and running:
1. Clone the repository 
2. Create a virtual environment (Computer_Vision01) 
3. Run the 2 scripts from the terminal

Output will be saved in the folder called Output


### 1. Clone the repository

The easiest way to access the files is to clone the repository from the command line using the following steps 

```bash
#clone repository as Image_Edge_Detection
git clone https://github.com/Orlz/Image-Edge-Detection.git Edge_detection

```


### 2. Create the virtual environment

You'll need to create a virtual environment which will allow you to run the script using all the relevant dependencies. This will require the requirements.txt file attached to this repository. 


To create the virtual environment you'll need to open your terminal and type the following code: 

```bash
bash create_virtual_environment.sh
```

And then activate the environment by typing: 
```bash
$ source Computer_Vision01/bin/activate
```


### 3. Run the Scripts

There are 2 scripts which can be run. The first contains a number of command-line parameters which you can be set by the user. The options for these are as follows: 

## edge_detection.py


Parameter options = 3

```
| Letter call   | Are             | Required?| Input Type   | Description
| ------------- |:-------------:  |:--------:|:-------------:
|`-i`           | `--image_path`  | No       | String       | Path to the image. Default: "Images/weholdtruths.jpeg"  |
|`-r`           | `--roi`         | Yes      | Integers     | Points of region of interest in image (need 4 integers) |
|`-o`           | `--output_path` | Yes      | String       | Path to the output directory                            |
```
        
Example of command line input

```bash 
$ python3 src/edge_detection.py --image_path <path-to-image> --roi x1 y1 x2 y2 --output_path <path-to-output-file>
```


Output
    - image_with_ROI.jpg: image from task 1 (ROI identified with a green rectange) 
    - image_cropped.jpg: image from task 2 (the cropped ROI image) 
    - image_letters.jpg: cropped image with contoured letters
  
  
    
Worked example of how command line code would look
    
``` bash
$ python3 src/edge_detection.py --roi 1400 890 2900 2800 --output_path Output/
```


Recommended edge inputs for ROI of Jefferson Monument 

x1 = 1400       x2 = 2900

y1 = 890        y2 = 2800



## ROI_detector.py

The ROI detector does not require any additional parameters to be included and can be run using the following code

``` bash
$ python3 src/ROI_detector.py 
```

Output:
6 images with green bounding boxes drawn around areas where the model detected text with enough certainty

The images can be found in the Output folder with the following name codes: "{location_name}_EAST.jpeg"



 
## Discussion of Results


### Part One

Conducting edge detection on our Jefferson Monument was considerably successful with every letter being detected and outlined with a green contour. Some of the surrounding border is also detected. This is an impressive result considering there are other lines and details in the image which have not been picked up, such as the outlines of the bricks. The image is however a good one to work with because it has a consistent background without any other objects distracting from the letters. Nevertheless, script demonstrates how a computer is able to detect and outline rather complex details within an image using the pixel intensities alone. Moreover, how it is able to do so in a small number of steps.   

### Part Two

The EAST text detector takes a completely different approach. It uses a complex neural network developed by the EAST team at Cornell University to look for patterns and identify where text might be. EAST is essentially a pretrained model which was developed using a vast array of images. Its purpose is to be able to find all kinds of text no matter how much detail is in the picture. The images it was trained on ranged from pictures of football player's on the picth with the name and number of their jersey being picked up, to the text on buses in a city scene, to advertisements within a shopping centre, and much more. More information on the model can be found [here](https://arxiv.org/abs/1704.03155v2) (Zhou, Xinyu & Yao, Cong & Wen, He & Wang, Yuzhi & Zhou, Shuchang & He, Weiran & Liang, Jiajun. (2017). EAST: An Efficient and Accurate Scene Text Detector.)
  


It is an interesting tool to use as a point of comparison to our more simple approach for a number of reasons. Firstly it introduces us to the problems which arise when images are not as good in quality as our Jefferson image is. Cosnider how the model struggles to manage the slanted writing on the Berlin wall (Berlin_EAST.jpeg). Secondly it demonstrates how sometimes it is the simple solutions which are the most effective. Consider how the EAST model struggles to identify the text on the Jefferson image at all, producing a number of somewhat random regions of interest. Here we should remember that what we are doing is comparing a model which looks for patterns to detect text, against a method which uses a more bottom up approach of pixel intensities across the whole image to look for text. The bottom up method does a much better job in most cases, with the more complex EAST model struggling as soon as the text is in a format it isn't used to (such as the block capital letters in the US image) or in an orientation which doesn't lie along the horizontal axis (such as in the Berlin image). However this perhaps leads us to the third point, that the comparison is a very good example of showing how important it is to chose a method of image processing which suits the images in question. The EAST model has been highly used in research and is not a useless model, yet in this context it struggles to handle the rugged style of our graffiti images. We see how accurate it can be in the Syrian picture (Syria_EAST.jpeg) and also the Northern Ireland image )N_Ireland_EAST.jpeg) - where it is accurate in detecting the text, but is too sensitive and not particularly specific, picking up the details of the gate also which are arguably not resemblent of text at all. 

Therefore, in conclusion, it seems that we have not been able to accurately identify a way to detect the text region of interest within an image. This may not be too much of a problem, since image processing has gone beyond these basic steps and has tools to process text in images effectively without needing to identify the region of interest, such as with Optical Character Recognition (OCR). However, as we see here, the complex methods come with their own limitations and should not overshadow the value of simple edge detection using kernals such as canny edge detection. Further considerations of this asssignment could try to implement some OCR techniques to pictures of graffiti and see how well these more tested neural networks could handle the complexities of graffiti text. 




