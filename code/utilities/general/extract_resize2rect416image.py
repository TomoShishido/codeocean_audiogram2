
import math
import cv2, os, copy
import glob
import numpy as np
import pandas as pd

def angle(pt1, pt2, pt0) -> float:
    dx1 = float(pt1[0,0] - pt0[0,0])
    dy1 = float(pt1[0,1] - pt0[0,1])
    dx2 = float(pt2[0,0] - pt0[0,0])
    dy2 = float(pt2[0,1] - pt0[0,1])
    v = math.sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) )
    return (dx1*dx2 + dy1*dy2)/ v

def findSquares2nd(bin_image, image, cond_area = 1000):
    
    rectangles =[]
    
    contours, _ = cv2.findContours(bin_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for i, cnt in enumerate(contours):
        
        arclen = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, arclen*0.02, True)

        area = abs(cv2.contourArea(approx))

        if approx.shape[0] == 4 and area > cond_area and cv2.isContourConvex(approx) :
            maxCosine = 0

            for j in range(2, 5):
                cosine = abs(angle(approx[j%4], approx[j-2], approx[j-1]))
                maxCosine = max(maxCosine, cosine)

            if maxCosine < 0.3 :
                rectangle ={'approx':approx,'area':area}
                rectangles.append(rectangle)

    rectangles_sorted = sorted(rectangles, key=lambda x:x['area'],reverse = True)            
                
    rcnt = rectangles_sorted[1]['approx'].reshape(-1,2)
    cv2.polylines(image, [rcnt], True, (0,0,255), thickness=2, lineType=cv2.LINE_8)
    return image, rectangles_sorted[1]

def givesquares2nd(image):
    if image is None :
        exit(1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    rimage, rectangle2nd_dicdata = findSquares2nd(bw, image)

    rectangle2nd_xyarray =rectangle2nd_dicdata['approx']
    return rectangle2nd_xyarray

def extract2ndrect_resize416x416(ORIGINAL_IMAGE_FILEPATH,REC2NDRESIZE416X416_SAVE_DIR):
    os.makedirs(REC2NDRESIZE416X416_SAVE_DIR, exist_ok=True)
    
    files = glob.glob(ORIGINAL_IMAGE_FILEPATH)
    for file in files:
        if file.endswith('jpg') or file.endswith('JPG'):
            dirname = os.path.dirname(file)
            namewithoutext = os.path.splitext(os.path.basename(file))[0]
            originalimg = cv2.imread(file, cv2.IMREAD_COLOR)
            img0 = originalimg.copy()
            rectangle2nd_xyarray = givesquares2nd(img0)

            top_bottom_array = []
            left_right_array = []
            top =bottom=left= right=0
            for each in rectangle2nd_xyarray:
                top_bottom_array.append(each[0][1])
                left_right_array.append(each[0][0])
                top=np.min(top_bottom_array)
                bottom=np.max(top_bottom_array)
                left = np.min(left_right_array)
                right = np.max(left_right_array)

            img1 = originalimg[top : bottom, left: right]
            img_resized = cv2.resize(img1,(416,416))

            img_filepath = REC2NDRESIZE416X416_SAVE_DIR + '/' + namewithoutext + '.jpg'
            cv2.imwrite(img_filepath, img_resized)
