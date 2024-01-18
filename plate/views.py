from django.shortcuts import render, redirect 
from django.http import HttpResponse
from .forms import *
import sys
from .models import uploadimage
import os 
import cv2, matplotlib, sys
import numpy as np
import operator
import matplotlib.pyplot as plt
from local_utils import detect_lp
from os.path import splitext, basename
from keras.models import model_from_json
import glob
import math
from PIL import Image, ImageOps, ImageChops
def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)
        
wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)

def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img


def get_plate(image_path, Dmax=608, Dmin=256):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return LpImg, cor


# Function to erase margins
def delete_margin(img):
    img2 = img.convert("RGB")
    bg = Image.new("RGB", img2.size, img2.getpixel((0, 0)))
    diff = ImageChops.difference(img2, bg)
    croprange = diff.convert("RGB").getbbox()
    nim = img.crop(croprange)
    return nim

# Function for joining images
def resize_connect(im1, im2, resample=Image.BICUBIC, resize_big_image=True):
    if im1.height == im2.height:
        _im1 = im1
        _im2 = im2
    elif (((im1.height > im2.height) and resize_big_image) or
          ((im1.height < im2.height) and not resize_big_image)):
        _im1 = im1.resize((int(im1.width * im2.height / im1.height), im2.height), resample=resample)
        _im2 = im2
    else:
        _im1 = im1
        _im2 = im2.resize((int(im2.width * im1.height / im2.height), im1.height), resample=resample)
    dst = Image.new('RGB', (_im1.width + _im2.width, _im1.height))
    dst.paste(_im1, (0, 0))
    dst.paste(_im2, (_im1.width, 0))
    return dst

############################################################################################################
# TrainAndTest.py

import cv2
import numpy as np
import operator
import os

# module level variables ##########################################################################
MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

###################################################################################################
class ContourWithData():

    # member variables ############################################################################
    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    def calculateRectTopLeftPointAndWidthAndHeight(self):               # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):                            # this is oversimplified, for a production grade program
        if self.fltArea < MIN_CONTOUR_AREA: return False        # much better validity checking would be necessary
        return True

###################################################################################################
def read_plate(image_name):
    img_basename = os.path.basename(image_name)
    img_basename=img_basename[:2]           #first two characer from basename
    allContoursWithData = []                # declare empty lists,
    validContoursWithData = []              # we will fill these shortly

    try:
        npaClassifications = np.loadtxt("trianingFile/Classifications"+img_basename+".txt", np.float32)                  # read in training classifications
    except:
        print("error, unable to open classifications.txt, exiting program\n")
        os.system("pause")
        return
    # end try

    try:
        npaFlattenedImages = np.loadtxt("trianingFile/FlattenedImages"+img_basename+".txt", np.float32)                 # read in training images
    except:
        print("error, unable to open flattened_images.txt, exiting program\n")
        os.system("pause")
        return
    # end try

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       # reshape numpy array to 1d, necessary to pass to call to train

    kNearest = cv2.ml.KNearest_create()                   # instantiate KNN object

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
    #pic="19"
    imgTestingNumbers = cv2.imread(image_name)          # read in testing numbers image

    if imgTestingNumbers is None:                           # if image was not read successfully
        print("error: image not read from file \n\n")        # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return                                              # and exit function (which exits program)
    # end if

    imgGray = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)       # get grayscale image
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                    # blur

                                                        # filter image from grayscale to black and white
    imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # input image
                                      255,                                  # make pixels that pass the threshold full white
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                                      cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                                      11,                                   # size of a pixel neighborhood used to calculate threshold value
                                      2)                                    # constant subtracted from the mean or weighted mean

    imgThreshCopy = imgThresh.copy()        # make a copy of the thresh image, this in necessary b/c findContours modifies the image

    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,             # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                 cv2.RETR_EXTERNAL,         # retrieve the outermost contours only
                                                 cv2.CHAIN_APPROX_SIMPLE)   # compress horizontal, vertical, and diagonal segments and leave only their end points

    for npaContour in npaContours:                             # for each contour
        contourWithData = ContourWithData()                                             # instantiate a contour with data object
        contourWithData.npaContour = npaContour                                         # assign contour to contour with data
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    # get bounding rect info
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
        allContoursWithData.append(contourWithData)                                     # add contour with data object to list of all contours with data
    # end for

    for contourWithData in allContoursWithData:                 # for all contours
        if contourWithData.checkIfContourIsValid():             # check if valid
            validContoursWithData.append(contourWithData)       # if so, append to valid contour list
        # end if
    # end for

    validContoursWithData.sort(key = operator.attrgetter("intRectX"))         # sort contours from left to right

    strFinalString = ""         # declare final string, this will have the final number sequence by the end of the program

    for contourWithData in validContoursWithData:            # for each contour
                                                # draw a green rect around the current char
        cv2.rectangle(imgTestingNumbers,                                        # draw rectangle on original testing image
                      (contourWithData.intRectX, contourWithData.intRectY),     # upper left corner
                      (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
                      (0, 255, 0),              # green
                      )                        # thickness

        imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,     # crop char out of threshold image
                           contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]

        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))             # resize image, this will be more consistent for recognition and storage

        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))      # flatten image into 1d numpy array

        npaROIResized = np.float32(npaROIResized)       # convert from 1d numpy array of ints to 1d numpy array of floats

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)     # call KNN function find_nearest

        strCurrentChar = str(chr(int(npaResults[0][0])))                                             # get character from results

        strFinalString = strFinalString + strCurrentChar            # append current char to full string
    # end for

    print("\n" + strFinalString + "\n") 
    return strFinalString

def upload_view(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
         
# Get the current instance object to display in the template
            img_obj = form.instance
            image_object=str(img_obj.upload_Img)
            upload_image='media/'+image_object
            LpImg,cor= get_plate(upload_image)


# Visualize our result
            plt.figure(figsize=(8,5))
            plt.axis(False)
            plt.imshow(LpImg[0])
            image_name=image_object
            plt.savefig(image_name,bbox_inches='tight',pad_inches = 0)

#crop the border of image to make one line
            img = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
            image_bicubic = cv2.resize(img, (1280,720), cv2.INTER_CUBIC)
            grayImage = cv2.cvtColor(image_bicubic, cv2.COLOR_BGR2GRAY)
            (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 117, 255, cv2.THRESH_BINARY)
            cv2.imwrite(image_name, blackAndWhiteImage)
            img = Image.open(image_name)
            border = (80, 47, 58, 40) # left, up, right, bottom
            ImageOps.crop(img, border).save(image_name)

            img = Image.open(image_name)
            pixels = img.load() 
# create the pixel map
            print(img.size[0])
            print(img.size[1])
            maxj=30
            count=0
            maxi=math.floor(img.size[1]*0.65)

            for i in range(maxi, img.size[1]): # for every pixel:
                for j in range(maxj):
                    if pixels[j,i] == 0:
                        count=count+1
            if count==0:
                border = (maxj, 0, 0, 0)
                ImageOps.crop(img,border).save(image_name)
            
                
#Load original image
            original_image = Image.open(image_name)
            width = original_image.size[0]
            height = original_image.size[1]

# Read the pixel-by-pixel color of the original image and convert from BGR to RGB
            pix_img = cv2.imread(image_name) 
            pix_img = cv2.cvtColor(pix_img, cv2.COLOR_BGR2RGB)

# Extract all white columns from between place names and numbers
            all_white = np.full((width ,3), 255)
            white_index = []
            for i in range(int(height*0.2), int(height*0.8)):
                if (pix_img[i]==all_white).all(axis=0).any() == True:
                    white_index.append(i)

# Stores the top and bottom columns of the white columns
            try:
                upper_white_index = min(white_index)
                lower_white_index = max(white_index)
# If all white columns do not exist → Suspended as unprocessable
            except ValueError:
                print(image_name + " Cannot be processed.")
                sys.exit()

# Top: Trimming place names
            box = (int(width*0.2), 1, int(width*0.8), upper_white_index)
            place_image = original_image.crop(box)
            place_image = delete_margin(place_image)

# Bottom: Hiragana (small), number trimming
            box = (1, lower_white_index, width, height)
            num_hiragana_image = original_image.crop(box)

# Trimming only hiragana
            bottom_height = height - lower_white_index
            box = (1, int(bottom_height*0.2), int(width*0.2), int(bottom_height*0.8))
            hiragana_image = num_hiragana_image.crop(box)
            hiragana_image = delete_margin(hiragana_image)

# Trim only the number
            box = (int(width*0.2), 1, width, bottom_height)
            number_image = num_hiragana_image.crop(box)
            number_image = delete_margin(number_image)

# Joining and saving images
            bottom_image = resize_connect(hiragana_image, number_image)
            resize_connect(place_image, bottom_image).save(image_name, quality=95)
#print(image_name + 'Image processing is finished.')

            img = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
            scale_percent = 40 # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
# resize image
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            cv2.imwrite(image_name, resized) 
            
            
            plate_number=read_plate(image_name)
            i=0
            plate=""
            for x in  plate_number:
                if plate_number.index(x)==i and i<=2 :
                    plate+=x
                i+=1
                if plate_number.index(x)>2:
                    plate+=x
            plate=plate.replace(' ','')
            
            return render(request, 'index.html', {'form': form, 'img_obj': img_obj, 'plate': plate,'upload':upload_image})
    else:
        form = UploadForm()
    return render(request, 'index.html', {'form': form})

###################################################################################################









