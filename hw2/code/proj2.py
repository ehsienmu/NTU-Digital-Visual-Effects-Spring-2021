#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
from scipy.ndimage import gaussian_filter
from numpy import unravel_index
import argparse
from tqdm import tqdm

def non_maxima_suppression(RT, window_size):
    pos = []
    feature_x = []
    feature_y = []
    [height,width] = RT.shape
    for i in range(5,width-window_size+1-5,window_size):
        for j in range(5,height-window_size+1-5,window_size):
            current_window = RT[j:j+window_size, i:i+window_size]
            row, col = unravel_index(current_window.argmax(), current_window.shape)
            max_val = current_window.max()
            if(max_val != -1):
                global_row = j + row
                global_col = i + col
                pos_tuple = (global_col,global_row)
                pos.append(pos_tuple)
    return pos

def harris_Corner_Detector(img, k, threshold, sigma):
    img_g = cv2.cvtColor(np.float32(img),cv2.COLOR_BGR2GRAY)
    I = gaussian_filter(img_g,sigma)
    [Ix,Iy] = np.gradient(I)
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    sigma_d = sigma
    Sx2 =  gaussian_filter(Ix2,sigma_d)
    Sy2 =  gaussian_filter(Iy2,sigma_d)
    Sxy =  gaussian_filter(Ixy,sigma_d)
    
    R = (Sx2 * Sy2 - Sxy * Sxy) - k * (Sx2 + Sy2) ** 2
    
    RT = np.where(R > threshold,R,-1)
    pos = non_maxima_suppression(RT, 10)
    return pos

def cylindrical(image, focal_length):
    y_origin = image.shape[0] // 2
    x_origin = image.shape[1] // 2
    new_image = np.zeros(image.shape,dtype = 'int')
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            x_cor = x - x_origin
            y_cor = y - y_origin
            x_new = round(focal_length * math.atan(x_cor/focal_length)) + x_origin
            y_new = round(focal_length * y_cor/(math.sqrt(x_cor*x_cor + focal_length*focal_length))) + y_origin
            new_image[y_new][x_new] = image[y][x]
    return new_image.astype(np.uint8)

def get_neighbor_img(img, x, y):
    kernel_size = 40
    w = img.shape[1]
    h = img.shape[0]
    kernel_size = min(kernel_size,x,y,abs(x-w),abs(y-h))
    return img[y-kernel_size:y+kernel_size,x-kernel_size:x+kernel_size][:][:]

def print_img(img, title='Matplotlib'):
    #Display image using matplotlib (Works)
    b,g,r = cv2.split(img)
    frame_rgb = cv2.merge((r,g,b))
    plt.imshow(frame_rgb)
    plt.title(title) 

def print_img_gray(img, title='Matplotlib'):
    #Display image using matplotlib (Works)
    g = cv2.split(img)
    frame_g = cv2.merge((g))
    plt.imshow(frame_g, cmap='gray')
    plt.title(title) 

def get_all_description(img, keypoints):
    des = []
    for kp in keypoints:
        #posx, posy= kp.pt
        posx, posy = kp
        little_img = get_neighbor_img(img,int(posx),int(posy)) # 90 * 90
        if(little_img.shape[0] != 80):
            continue
        if len(little_img) > 0:
            mini_img = cv2.resize(little_img,(10, 10)) # blur to 10, 10
            flat_gray = (cv2.cvtColor(mini_img,cv2.COLOR_BGR2GRAY)).flatten()
            des.append(((int(posx),int(posy)), flat_gray.astype(np.float32)))
    return des




def match_features(pic1, pic2, threshold = 800):
    # pic1, pic2 is the feature description list, ex: pic1 = [description_feature_1, description_feature_2, description_feature_3, ...]
    # description_feature_1 is a col vector with dim = 100
    # find min(mse) for each feature in pic1
    match_result = []
    for i in range(len(pic1)):
        min_mse = 10000
        best_match = -1
        for j in range(len(pic2)):
            mse = ((pic1[i][1] - pic2[j][1])**2).mean(axis=None)
            if(mse < min_mse):
                min_mse = mse
                best_match = j
        if min_mse <= threshold:
            match_result.append((pic1[i][0], pic2[best_match][0]))
    # match_result_format = [(img1_feature, img2_feature), (, ), ...]
    return match_result

def ransac_best_moving(match_result, voting_threshold = 3):# voting_threshold: square_distance
    moving_cadidate = list(map(lambda x: (x[0][0] - x[1][0], x[0][1] - x[1][1]), match_result))
    voting_result = []
    for i in range(len(moving_cadidate)):
        cadidate = moving_cadidate[i]
        all_voting_pair = list(map(lambda x: (x[0] - cadidate[0], x[1] - cadidate[1]), moving_cadidate))
        distance_square = list(map(lambda x: x[0] * x[0] + x[1] * x[1], all_voting_pair))
        vote_yes = (np.count_nonzero(np.array(distance_square) < voting_threshold))
        voting_result.append(vote_yes)
    best_moving_index = (np.argmax(np.array(voting_result)))
    best_moving = moving_cadidate[best_moving_index]
    best_match = match_result[best_moving_index]
    return best_moving, best_match

def get_move(img1, img2, detect_threshold = 4000, match_threshold = 200, voting_threshold = 3):
    img1 = img1.astype('uint8')
    img2 = img2.astype('uint8')
    # detect for two Images
    keypoints_1 = harris_Corner_Detector(img1, k = 0.06, threshold = detect_threshold, sigma = 0.5)    
    keypoints_2 = harris_Corner_Detector(img2, k = 0.06, threshold = detect_threshold, sigma = 0.5)
    img1_corner = img1
    img2_corner = img2

    # for each_corner in range(len(keypoints_1)):
    #     cv2.circle(img1_corner, keypoints_1[each_corner], 2, (0,0,255), -1)
    #     cv2.imwrite('output/resize_lake_corners_{}.jpg'.format(i), img1)
    # for each_corner in range(len(keypoints_2)):
    #     cv2.circle(img2_corner, keypoints_2[each_corner], 2, (0,0,255), -1)
    #     cv2.imwrite('output/resize_lake_corners_{}.jpg'.format(i+0.5), img2)
        
    # Descripte feature
    descript_img1 = get_all_description(img1, keypoints_1)
    descript_img2 = get_all_description(img2, keypoints_2)
    
    # Match feature
    match_result = match_features(descript_img1, descript_img2, match_threshold)
    
    # RANSAC get best moving
    best_moving, best_match = ransac_best_moving(match_result, voting_threshold)
    return best_moving, best_match

def padding_img(img, move_x, move_y):
    if(move_x >= 0 and move_y >= 0):
        new_img = np.pad(img, ((move_y, 0), (move_x, 0), (0, 0)), 'constant')
    elif(move_x >= 0 and move_y < 0):
        new_img = np.pad(img, ((0, -move_y), (move_x, 0), (0, 0)), 'constant')
    elif(move_x < 0 and move_y >= 0):
        new_img = np.pad(img, ((move_y, 0), (0, -move_x), (0, 0)), 'constant')
    else:
        new_img = np.pad(img, ((0, -move_y), (0, -move_x), (0, 0)), 'constant')
    return new_img

def stiching_img(best_moving, best_match, src1, src2, previous_movex = 0):
    move_x, move_y = best_moving
    if(move_x < 0):
        move_x = -move_x
        move_y = -move_y
        best_match = (best_match[1], best_match[0])
        src1, src2 = src2, src1
    src1_padding_x = src2.shape[1] - src1.shape[1] + best_match[0][0] - best_match[1][0] 
    src2_padding_x = best_match[0][0] - best_match[1][0] 
    intersect_range = best_match[1][0] - best_match[0][0] + src1.shape[1]
    new_img1 = padding_img(src1, -src1_padding_x, -move_y)
    new_img2 = padding_img(src2, src2_padding_x, move_y)
    final_pic = (np.zeros(new_img1.shape,'int'))
    intersect_cnt = 0
    for i in range(0, final_pic.shape[1]):

        origin_percent = 1
        new_percent = 1

        if (np.count_nonzero(new_img1[:,i][:] != 0)) and (np.count_nonzero(new_img2[:,i][:] != 0)):
            origin_percent = (1 - intersect_cnt / intersect_range)
            new_percent = (intersect_cnt / intersect_range)
            intersect_cnt += 1
        elif(np.count_nonzero(new_img1[:,i][:] != 0)):
            origin_percent = 1
            new_percent = 0
        elif(np.count_nonzero(new_img2[:,i][:] != 0)):
            origin_percent = 0
            new_percent = 1

        final_pic[:,i][:] = origin_percent * new_img1[:,i][:]  + new_percent* new_img2[:,i][:]  
        
    return final_pic

def getInfo(photo_info_path):
    src_list = []
    focal_length_list = []
    with open(photo_info_path, 'r') as f:
        lines = f.read().splitlines() 
        for line in lines:
            if (line.find(' ') == -1 and len(line) != 0):
                if (line.find('.jpg') != -1) or (line.find('.png') != -1):
                    #print(line)         
                    if(line.rfind('\\') != -1):  
                        src_list.append(line[line.rfind('\\') + 1:])
                    else:
                        src_list.append(line)

                else:
                    focal_length_list.append(float(line))

    return src_list, focal_length_list

#main function
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'pano')

    parser.add_argument('-photo_path', help='photo path', dest='photo_path', default='cwd')
    parser.add_argument('-focal_filename', help='focal_length.txt filename, ex: pano.txt', dest='focal_filename', default='pano.txt')
    parser.add_argument('-detect_threshold', help='detect feature threshold', dest='detect_threshold', default='50000')
    parser.add_argument('-match_threshold', help='match threshold', dest='match_threshold', default='400')
    args = parser.parse_args()

    if(args.photo_path):
        photo_dir_path = args.photo_path

    if(photo_dir_path == 'cwd'):
        photo_dir_path = os.getcwd()+'/'
    else:
        if(photo_dir_path[-1]!='/'):
            photo_dir_path = photo_dir_path + '/'

    photo_info_path = ""
    if((args.focal_filename.find('/') == -1) and (args.focal_filename.find('\\') == -1)):
        photo_info_path = photo_dir_path + args.focal_filename
    else:
        photo_info_path = args.focal_filename

    src_list, focal_length_list = getInfo(photo_info_path)

    # print('src_list = \n', src_list)
    # print('focal_length_list = \n', focal_length_list)

    #initail the image as image 1
    src = cv2.imread(photo_dir_path + src_list[0])
    if src is None:
        print('Could not open or find the image:', (src_list[0]))

    focal_src = focal_length_list[0]

    # make cylindrical
    final_img = cylindrical(src, focal_src)

    total_image_cnt = len(src_list)
    for i in tqdm(range(1, total_image_cnt),desc='Stitching image: ', ascii=True):
        src1_path = photo_dir_path + src_list[i]
        src1 = cv2.imread(src1_path)
        if src1 is None:
            print('Could not open or find the image:', (src1_path))
          
        focal_src_1 = focal_length_list[i]
        # make cylindrical
        src1 = cylindrical(src1, focal_src_1)
        #print("shape",final_img.shape," ",src1.shape)
        dy = final_img.shape[0] - src1.shape[0]
        src1 = padding_img(src1, 0, dy)
        # cv2.imwrite("output/src1{}.jpg".format(i), src1)
        best_moving, best_match = get_move(final_img, src1, detect_threshold = int(args.detect_threshold), match_threshold = int(args.match_threshold))
        #print("best_moving",best_moving)
        #print("best_match",best_match)
        final_img = stiching_img(best_moving, best_match, final_img, src1)
    #print_img(final_img, 'Panorama')

    #print(final_img.shape[0])

    # cut the black row
    black_row_count = 0
    for i in range(0, final_img.shape[0]):
        if (np.count_nonzero(final_img[i,:][:] != 0)):
            break
        else:
            black_row_count += 1
    #print(black_row_count)

    pano_img = final_img[black_row_count:,:,:]

    #print_img(pano_img)

    cv2.imwrite(photo_dir_path + "panoroma.jpg", pano_img)
