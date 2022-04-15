# import library 
import argparse,sys
from pickle import FALSE, TRUE
import uuid
import os
from os import listdir, mkdir, chdir, rename
from os.path import isfile, join, isdir, exists, basename
import numpy as np
from scipy.ndimage import zoom
import cv2
import tempfile
import shutil
import glob
import re
import csv  
import math
from math import sqrt
from scipy.spatial import ConvexHull
from scipy.spatial import distance as dist
from PIL import Image
from difflib import SequenceMatcher
import pandas as pd
import json

#Import pytesseract
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


#Import  Craft library
# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from skimage import io
import craft_utils
import imgproc
import file_utils
import json
import zipfile
from craft import CRAFT
from collections import OrderedDict



# https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions/37121993#37121993
def clipped_zoom(img, zoom_factor, **kwargs):
    h, w = img.shape[:2]
    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)
    # Zooming out
    if zoom_factor < 1:
        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        # Zero-padding
        #out = np.zeros_like(img)
        # White-padding
        out=np.full_like(img,255)
        #out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)
        out[top:top+zh, left:left+zw]=cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)
    # Zooming in
    elif zoom_factor > 1:
        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)
        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]
    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out

# https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return buf

def dilate_char(img, kernelsize=3, shape=cv2.MORPH_ELLIPSE):
    kernel = cv2.getStructuringElement(shape,(kernelsize,kernelsize))
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    dilation = cv2.dilate(opening,kernel,iterations = 2)
    return (255-dilation)


def preprocess_CRAFT(img):
    final=dilate_char(255-img,4)
    return final
    
    
# https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/

def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="int32")

# https://gis.stackexchange.com/questions/22895/finding-minimum-area-rectangle-for-given-points
def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
#     rotations = np.vstack([
#         np.cos(angles),
#         -np.sin(angles),
#         np.sin(angles),
#         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval

# Rotation image: https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
def rotate_image(image, angle, center):
    image_center = tuple(np.array(center).astype(float))
    #print(image_center, angle, 1.0)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], borderValue=(255,255,255))
    return result


#_________________________________________
#____________________CRAFT_FUNCTION_______
#_________________________________________
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

##___CRAFT___TEST_NET___________________
def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio
    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()
    # forward pass
    with torch.no_grad():
        y, feature = net(x)
    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()
    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()
    t0 = time.time() - t0
    t1 = time.time()
    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]
    t1 = time.time() - t1
    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)
    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))
    return boxes, polys, ret_score_text
##________________________________________



#Argument folder 
parser = argparse.ArgumentParser(description='OCR-Dotted-Matrix')
parser.add_argument('--image', type=str, help='test image')
parser.add_argument('--label', type=str, help='label')
parser.add_argument('--folder_res', type=str, help='folder result')


#Argument CRAFT
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.9, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.3, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.1, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=True, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
args = parser.parse_args()




if __name__=="__main__":


    #____________________FOLDER_INITIALIZE____________
    if args.image is None:
        print("insert test path image...")
        sys.exit()
    image=args.image
    if not exists(image):
        print("the specified path does not exist...")
        sys.exit()
    if not isfile(image):
        print("The item at the specified path does not appear to be a file..")
        sys.exit()
    #SAVE_folder_result  
    try:
        mkdir("result")
    except FileExistsError:
    # directory already exists
        pass  
    save_path_folder=join("result",args.folder_res)
    try:
        mkdir(save_path_folder)
    except FileExistsError:
    # directory already exists
        pass
    

        

    headers = ['Name_original_file',
                'Name_preprocess', 
                'check_label', 
                'tesseract_LCDDot_FT_500_psm3_result',
                'LCDDot_FT_500_psm3_sequence_matcher_ratio_result',
                'LCDDot_FT_500_psm3_bool_re_result']

    #_______________________________
    #____LOADING_NET______________
    #_______________________________
    net = CRAFT()     # initialize
    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))
    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False
    net.eval()
    #_______________________________
    # ____LINK REFINER______________
    #_______________________________
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        #print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()


    #_______________________________
    # ____TEST_IMAGES______________
    #_______________________________
    #for filename in filename_array:
    start_time = time.time()

    check_label=args.label
    filename=os.path.basename(args.image)

    save_path=join(save_path_folder,filename)
    save_path_CRAFT=join(save_path,"result_CRAFT")
    save_path_images=join(save_path,"ritagli")


    
    #_CREATE folder for single image test
    try:
        mkdir(save_path)
        mkdir(save_path_CRAFT)
        mkdir(save_path_images)
    except FileExistsError:
    # directory already exists
        pass

    img=cv2.imread(args.image,0)
    image_basename=basename(filename)
    zoomed=clipped_zoom(img,0.7) 

    cv2.imwrite(join(save_path,"zoomed.jpg"),zoomed)
    preprocessed=preprocess_CRAFT(zoomed)
    cv2.imwrite(join(save_path,"preprocessed.jpg"),preprocessed)


    image = imgproc.loadImage(join(save_path,"preprocessed.jpg"))
    bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)

    # save score text
    mask_file = save_path_CRAFT + "/res_preprocessed_mask.jpg"
    cv2.imwrite(mask_file, score_text)

    file_utils.saveResult("preprocessed", image[:,:,::-1], polys, dirname=save_path_CRAFT)

    

   #______________________________________________________
   #__________EXTRACT_THE_BOUNDING_BOX____________________
   #______________________________________________________
    with open(join(save_path_CRAFT,"res_preprocessed.txt")) as fin:
        i=0
        for line in fin:
            if(len(line)!=1):
                coord=np.array(line.rstrip().split(',')).astype(int).reshape(-1,2)
                if len(coord)>4:
                    pts=minimum_bounding_rectangle(coord)
                    coord=order_points(pts)
                diffs=[coord[0,0]-coord[-1,0],coord[0,1]-coord[-1,1]]
                angle=-(math.atan(diffs[0]/diffs[1])*180/math.pi)
                dist1=math.ceil(sqrt((coord[-1,0]-coord[0,0])**2+(coord[-1,1]-coord[0,1])**2))
                dist2=math.ceil(sqrt((coord[1,0]-coord[0,0])**2+(coord[1,1]-coord[0,1])**2))
                im=zoomed
                center=[coord[0,0],coord[0,1]]
                rotated=rotate_image(im,angle,center)
                cut=rotated[coord[0,1]:coord[0,1]+dist1,coord[0,0]:coord[0,0]+dist2]
                if cut.shape[0]>cut.shape[1]:
                    cut=cv2.rotate(cut,0)

                ###### Original image (without preprocessing) 
                im_pil = Image.fromarray(cut)
                im_pil.save(join(save_path_images,'_original_'+'{}'.format(i)+'.jpg'), dpi=(300.0, 300.0))
                cut_flipped=cv2.rotate(cut,1)

                ####### PREPROCESSING
                #  try different parameters of resize and kernel of Morphological Transformations
                #  for enhancing tesseract  recognition                          
                size_pool=((1.2,1.2),(1.5,1.5))
                kernel_pool=((3,3),(5,5),(7,7),(9,9))
                clipped_zoom_pool=(0.3,0.4,0.5,0.7)
                kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) #kernel per
                kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,5)) #kernel per

                index_save=0
                for fx,fy in size_pool:
                    image_original=cv2.resize(cut, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
                    for factor in clipped_zoom_pool:
                            image_original_resize=clipped_zoom(image_original,factor)
                            _, blackAndWhite = cv2.threshold(image_original_resize,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                            for kernel in kernel_pool:
                                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
                                opening_1= cv2.morphologyEx(blackAndWhite, cv2.MORPH_OPEN, kernel)
                                # Saving
                                im_pil = Image.fromarray(opening_1)
                                im_pil.save(join(save_path_images,'_preprocess_'+'{}{}'.format(index_save,i)+'.jpg'), dpi=(300.0, 300.0))
                                opening_2 = cv2.dilate(255-opening_1, kernel2,iterations = 1) 
                                im_pil = Image.fromarray(255-opening_2)
                                im_pil.save(join(save_path_images,'_preprocess_kernel2'+'{}{}'.format(index_save,i)+'.jpg'), dpi=(300.0, 300.0))
                                opening_3 = cv2.dilate(255-opening_1, kernel2,iterations = 1) 
                                im_pil = Image.fromarray(255-opening_3)
                                im_pil.save(join(save_path_images,'_preprocess_kernel3'+'{}{}'.format(index_save,i)+'.jpg'), dpi=(300.0, 300.0))
                                index_save+=1
            i+=1

                ###### 
        if i==0:
            print("No text was detected (no bounding box) in the image...")
    #_________________________________________________________
    #__________TEXT_RECOGNITION_WITH_TESSERACT__________________
    #_______________________________________________________

    if not listdir(save_path_images): 
        df = pd.DataFrame([[filename,'NONE', check_label.replace(" ", ""),"NONE",0.0,False]], columns=headers)
        with open(join(save_path,'result.json'), 'w') as the_file:
            the_file.write(df.to_json(orient="records"))
    else:
    
        files=[f for f in listdir(save_path_images) if isfile(join(save_path_images,f))]
        json_dict={}
        Name_array_preprocess=[]
        LCDDot_FT_500_psm3_ocr=[]
        LCDDot_FT_500_psm3_sequence_matcher_ratio=[]
        LCDDot_FT_500_psm3_bool_re=[]

        #tessaract ocr
        for file in files:
            img=cv2.imread(join(save_path_images,file),0)
            Name_array_preprocess.append(file)
            text_3=pytesseract.image_to_string(img, lang='LCDDot_FT_500', config="-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPRSTUVWXY.-:").replace('\x0c','').replace(' ','').replace("\n", " ").replace(" ", "") 
            LCDDot_FT_500_psm3_ocr.append(text_3)
            LCDDot_FT_500_psm3_sequence_matcher_ratio.append(round(SequenceMatcher(None, check_label.replace(" ", ""), text_3).ratio(), 2)) #ratio seqence matcher
            LCDDot_FT_500_psm3_bool_re.append(bool(re.match(check_label.replace(" ", ""),text_3))) #matche with re

        
        
        #create data frame to store result  of different preprocess
        d = {'Name_original_file': filename,
            'Name_preprocess': Name_array_preprocess,
            'check_label': check_label.replace(" ", ""),
            'tesseract_LCDDot_FT_500_psm3_result':LCDDot_FT_500_psm3_ocr,
            'LCDDot_FT_500_psm3_sequence_matcher_ratio_result':LCDDot_FT_500_psm3_sequence_matcher_ratio, 
            'LCDDot_FT_500_psm3_bool_re_result': LCDDot_FT_500_psm3_bool_re,
        }

        
        df = pd.DataFrame(data=d)
        dfj = df.groupby(["Name_preprocess"]).apply(lambda x: x.to_dict("r")).to_json(orient="records")
        #save dataframe in JSON file
        with open(join(save_path,'result.json'), 'w') as the_file:
            the_file.write(dfj)

    print("--- %s seconds ---" % (time.time() - start_time))






















    
      




            

       

        
        

