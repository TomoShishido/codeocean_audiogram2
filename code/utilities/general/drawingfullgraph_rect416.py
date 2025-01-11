import cv2, os
import glob
import numpy as np
import pandas as pd

def lineList(x1, y1, x2, y2): 
    line_lst = []
    step = 0
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    if dx > dy:
        if x1 > x2 :
            step = 0
            if y1 > y2:
                step = 1
            else:
                step = -1
            x1, x2 = x2, x1 # swap
            y1 = y2
        else:
            if y1 < y2:
                step = 1
            else:
                step = -1
        line_lst.append((x1, y1))
        s = dx >> 1
        x1 += 1
        while (x1 <= x2):
            s -= dy
            if s < 0:
                s += dx
                y1 += step
            line_lst.append((x1, y1))
            x1 += 1
    else:
        if y1 > y2:
            if x1 > x2:
                step = 1
            else:
                step = -1
           
            y1, y2 = y2, y1 # swap
            x1 = x2
        else:
            if x1 < x2:
                step = 1
            else:
                step = -1
        line_lst.append((x1, y1))
        s = dy >> 1
        y1 += 1
        while y1 <= y2:
            s -= dx
            if s < 0:
                s += dy
                x1 += step
            line_lst.append((x1, y1))
            y1 += 1
    return  line_lst



def drawDashedLine(img, start_point, end_point, gap, linewidth, color):
    li = lineList(start_point[0], start_point[1], end_point[0], end_point[1])
    fwd = start_point
    bwd = start_point
    j = 0
    for i, pt in enumerate(li):
        if i % gap == 0:
            bwd = pt

            if(j % 2):
                cv2.line(img, fwd, bwd, color, linewidth, lineType=cv2.LINE_AA)
            fwd = bwd
            j += 1
    return img

def RL_fulllinedot_aaplot(yololabeltxtfile, img_width, img_height):
    #imageArray = np.zeros((height, width, 3), np.uint8)
    img = np.zeros((416, 416, 3), np.uint8)#black image 416x416 (yoloimage)
    img += 255 #white image
    image_frame_height_factor = 1.0
    image_frame_width_factor = 1.0
    frame_top = 0
    frame_left = 0
    R_ac_list=[]
    R_bc_list=[]
    L_ac_list=[]
    L_bc_list=[]   
    with open(yololabeltxtfile) as f:
        txtlines = f.readlines()
    #draw a frame

    # set a processed area roi(left(x1), top(y1), right(x2), bottom(y2))
    left = 0#int((float(target_info[1]) - float(target_info[3]) / 2)*img_width)
    top = 0#int((float(target_info[2]) - float(target_info[4]) / 2)*img_height)
    right = 416#int((float(target_info[1]) + float(target_info[3]) / 2)*img_width)
    bottom = 416#int((float(target_info[2]) + float(target_info[4]) / 2)*img_height)
    width = right - left
    height = bottom - top
    image_frame_height_factor = img_height/height
    image_frame_width_factor = img_width/width
    frame_top = top
    frame_left = left
    cv2.rectangle(img, (0, 0), (416, 416), (0, 0, 0), 2)
    #draw 0db level
    for i in range(16):
        level_vertical_line = int((float(height)*(i/16))*image_frame_height_factor+13)
        if i==2:
            cv2.line(img, (0, level_vertical_line), (416, level_vertical_line), (0, 0, 0), 2)
        else:
            cv2.line(img, (0, level_vertical_line), (416, level_vertical_line), (0, 0, 0), 1)

    # level0db = int((float(height)*0.1538)*image_frame_height_factor)
    # cv2.line(img, (0, level0db), (416, level0db), (0, 0, 0), 2)
    
    #right red air conduction
    for textline in txtlines:
        target_info = textline.split() #target_info =[label, x, y, w, h]
        if target_info[0] == '0':# in the case of right air conduction
            # set a processed area roi(left(x1), top(y1), right(x2), bottom(y2))
            point_left = int((float(target_info[1]) - float(target_info[3]) / 2)*img_width)
            left = int((point_left-frame_left)*image_frame_width_factor)
            point_top = int((float(target_info[2]) - float(target_info[4]) / 2)*img_height)
            top = int((point_top-frame_top)*image_frame_height_factor)
            point_right =int((float(target_info[1]) + float(target_info[3]) / 2)*img_width)
            right = int((point_right-frame_left)*image_frame_width_factor)
            point_bottom = int((float(target_info[2]) + float(target_info[4]) / 2)*img_height)
            bottom = int((point_bottom-frame_top)*image_frame_height_factor)
            width = right - left
            height = bottom - top
            center = (int(float(left) +float(width)/2), int(float(top) +float(height)/2))
            radius = int(float(0.8*height)/2)
            #draw vertical line
            cv2.line(img, (int(float(left) +float(width)/2), 0), (int(float(left) +float(width)/2), 416), (0, 0, 0), 1)
            cv2.circle(img, center, radius, (0, 0, 255), 1)
            R_ac_dic={'center_x':int(float(left) +float(width)/2), 'center_y':int(float(top) +float(height)/2)}
            R_ac_list.append(R_ac_dic)
    #right red bone conduction
    for textline in txtlines:
        target_info = textline.split() #target_info =[label, x, y, w, h]
        if target_info[0] == '1':# in the case of right bone conduction
            # set a processed area roi(left(x1), top(y1), right(x2), bottom(y2))
            point_left = int((float(target_info[1]) - float(target_info[3]) / 2)*img_width)
            left = int((point_left-frame_left)*image_frame_width_factor)
            point_top = int((float(target_info[2]) - float(target_info[4]) / 2)*img_height)
            top = int((point_top-frame_top)*image_frame_height_factor)
            point_right =int((float(target_info[1]) + float(target_info[3]) / 2)*img_width)
            right = int((point_right-frame_left)*image_frame_width_factor)
            point_bottom = int((float(target_info[2]) + float(target_info[4]) / 2)*img_height)
            bottom = int((point_bottom-frame_top)*image_frame_height_factor)
            width = right - left
            height = bottom - top
            center = (int(float(left) +float(width)/2), int(float(top) +float(height)/2))
            # cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 5)
            cv2.putText(img,
                text='[',
                org=center,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 0, 255),
                thickness=1,
                lineType=cv2.LINE_4)
            R_bc_dic={'center_x':int(float(left) +float(width)/2), 'center_y':int(float(top) +float(height)/2)}
            R_bc_list.append(R_bc_dic)
    #left red air conduction
    for textline in txtlines:
        target_info = textline.split() #target_info =[label, x, y, w, h]
        if target_info[0] == '2':# in the case of right air conduction
            # set a processed area roi(left(x1), top(y1), right(x2), bottom(y2))
            point_left = int((float(target_info[1]) - float(target_info[3]) / 2)*img_width)
            left = int((point_left-frame_left)*image_frame_width_factor)
            point_top = int((float(target_info[2]) - float(target_info[4]) / 2)*img_height)
            top = int((point_top-frame_top)*image_frame_height_factor)
            point_right =int((float(target_info[1]) + float(target_info[3]) / 2)*img_width)
            right = int((point_right-frame_left)*image_frame_width_factor)
            point_bottom = int((float(target_info[2]) + float(target_info[4]) / 2)*img_height)
            bottom = int((point_bottom-frame_top)*image_frame_height_factor)
            width = right - left
            height = bottom - top
            center = (int(float(left) +float(width)/2), int(float(top) +float(height)/2))
            center_corrected = (int(float(left) +float(width)/2 -8), int(float(top) +float(height)/2 +8))
            
            radius = int(float(height)/2)
            # cv2.circle(img, center, radius, (255, 0, 0), 1)
            cv2.putText(img,
                text='X',
                org=center_corrected,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(255, 0, 0),
                thickness=1,
                lineType=cv2.LINE_4)
            L_ac_dic={'center_x':int(float(left) +float(width)/2), 'center_y':int(float(top) +float(height)/2)}
            L_ac_list.append(L_ac_dic)
    #right red bone conduction
    for textline in txtlines:
        target_info = textline.split() #target_info =[label, x, y, w, h]
        if target_info[0] == '3':# in the case of right bone conduction
            # set a processed area roi(left(x1), top(y1), right(x2), bottom(y2))
            point_left = int((float(target_info[1]) - float(target_info[3]) / 2)*img_width)
            left = int((point_left-frame_left)*image_frame_width_factor)
            point_top = int((float(target_info[2]) - float(target_info[4]) / 2)*img_height)
            top = int((point_top-frame_top)*image_frame_height_factor)
            point_right =int((float(target_info[1]) + float(target_info[3]) / 2)*img_width)
            right = int((point_right-frame_left)*image_frame_width_factor)
            point_bottom = int((float(target_info[2]) + float(target_info[4]) / 2)*img_height)
            bottom = int((point_bottom-frame_top)*image_frame_height_factor)
            width = right - left
            height = bottom - top
            center = (int(float(left) +float(width)/2 -8), int(float(top) +float(height)/2))
            
            # cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 5)
            cv2.putText(img,
                text=']',
                org=center,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 0, 0),
                thickness=1,
                lineType=cv2.LINE_4)
            L_bc_dic={'center_x':int(float(left) +float(width)/2), 'center_y':int(float(top) +float(height)/2)}
            L_bc_list.append(L_bc_dic)

    #overlapped air conduction
    for textline in txtlines:
        target_info = textline.split() #target_info =[label, x, y, w, h]
        if target_info[0] == '4':# in the case of overlapping right and left air conductions
            # set a processed area roi(left(x1), top(y1), right(x2), bottom(y2))
            point_left = int((float(target_info[1]) - float(target_info[3]) / 2)*img_width)
            left = int((point_left-frame_left)*image_frame_width_factor)
            point_top = int((float(target_info[2]) - float(target_info[4]) / 2)*img_height)
            top = int((point_top-frame_top)*image_frame_height_factor)
            point_right =int((float(target_info[1]) + float(target_info[3]) / 2)*img_width)
            right = int((point_right-frame_left)*image_frame_width_factor)
            point_bottom = int((float(target_info[2]) + float(target_info[4]) / 2)*img_height)
            bottom = int((point_bottom-frame_top)*image_frame_height_factor)
            width = right - left
            height = bottom - top
            center = (int(float(left) +float(width)/2), int(float(top) +float(height)/2))
            center_corrected = (int(float(left) +float(width)/2 -8), int(float(top) +float(height)/2 +8))
            radius = int(float(0.8*height)/2)
            #draw vertical line
            cv2.line(img, (int(float(left) +float(width)/2), 0), (int(float(left) +float(width)/2), 416), (0, 0, 0), 1)
            
            cv2.circle(img, center, radius, (0, 0, 255), 1)
            cv2.putText(img,
                text='X',
                org=center_corrected,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(255, 0, 0),
                thickness=1,
                lineType=cv2.LINE_4)
            M_ac_dic={'center_x':int(float(left) +float(width)/2), 'center_y':int(float(top) +float(height)/2)}
            R_ac_list.append(M_ac_dic)
            L_ac_list.append(M_ac_dic)
    #draw lines
    R_ac_sorted = sorted(R_ac_list, key=lambda x: x['center_x'])
    R_bc_sorted = sorted(R_bc_list, key=lambda x: x['center_x'])
    L_ac_sorted = sorted(L_ac_list, key=lambda x: x['center_x'])
    L_bc_sorted = sorted(L_bc_list, key=lambda x: x['center_x'])
    count = len(R_ac_sorted)
    if count > 1:
        for i in range(count-1):
            cv2.line(img, (R_ac_sorted[i]['center_x'], R_ac_sorted[i]['center_y']), (R_ac_sorted[i+1]['center_x'], R_ac_sorted[i+1]['center_y']), (0, 0, 255), 1)  
    count = len(L_ac_sorted)
    if count > 1:
        for i in range(count-1):
            # cv2.line(img, (L_ac_sorted[i]['center_x'], L_ac_sorted[i]['center_y']), (L_ac_sorted[i+1]['center_x'], L_ac_sorted[i+1]['center_y']), (255, 0, 0), 1)  
            drawDashedLine(img, (L_ac_sorted[i]['center_x'], L_ac_sorted[i]['center_y']), (L_ac_sorted[i+1]['center_x'], L_ac_sorted[i+1]['center_y']), gap=5, linewidth=1, color=(255, 0, 0))
    return img

def draw_RL_full_linedotgraph(YOLOIMG_FILE_PATH, FULLGRAPH_SAVE_DIR): 

    if not os.path.exists(FULLGRAPH_SAVE_DIR):
        os.mkdir(FULLGRAPH_SAVE_DIR)

    files = glob.glob(YOLOIMG_FILE_PATH)
    #To skip .txt files
    for file in files:
        if file.endswith('txt'):
            print('there is a text file.')
        else:
            dirname = os.path.dirname(file)
            namewithoutext = os.path.splitext(os.path.basename(file))[0]
            image_ext = os.path.splitext(os.path.basename(file))[1]
            yololabeltxtfile_PATH = dirname + '/labels/' + namewithoutext + '.txt'              

            img = cv2.imread(file)
            yoloimg=cv2.resize(img,(416,416))
            yoloimg_height, yoloimg_width = yoloimg.shape[:2]

            #Right dot graph
            RL_fulllinedot_aaplot_img = RL_fulllinedot_aaplot(yololabeltxtfile_PATH, yoloimg_width, yoloimg_height)
            # RL_fulllinedot_aaplot_img = RL_fulllinedot_aaplot(yololabeltxtfile_PATH, 210, 416)


            img_filepath = FULLGRAPH_SAVE_DIR + '/' + namewithoutext + image_ext
            cv2.imwrite(img_filepath, RL_fulllinedot_aaplot_img)