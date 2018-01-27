# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 20:43:57 2017

@author: Instrumentation
"""

import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
from scipy.ndimage.measurements import label
from functions import color_hist,bin_spatial,get_hog_features,slide_window,draw_boxes,search_windows

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, x_start_stop, y_start_stop, scale,color_space ,svc, X_scaler, orient, 
              pix_per_cell, cell_per_block, spatial_size, hist_bins, show_all_rectangles): 
    img = img.astype(np.float32)/255   
    box_list = []

    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
       
    img_tosearch = img[y_start_stop[0]:y_start_stop[1],x_start_stop[0]:x_start_stop[1],:]
#    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    if color_space != 'RGB':
        if color_space == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
        #ctrans_tosearch = ctrans_tosearch.astype(np.float32)/255
    else: ctrans_tosearch = np.copy(img_tosearch)      

    #print(ctrans_tosearch.shape)
    if scale != 1:
        imshape = ctrans_tosearch.shape       
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    #nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    if hog_channel == 'ALL':
            hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
            hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
            hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    else:
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            if hog_channel == 'ALL':
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_features = hog_feat1
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            features = np.hstack((spatial_features, hist_features, hog_features))
            #print(np.shape(features))
            # Scale features and make a prediction
            features[np.isnan(features) == True] = 0
            test_features = X_scaler.transform(np.array(features).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((hog_features)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1 or show_all_rectangles:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                box_list.append([(xbox_left+x_start_stop[0], ytop_draw+y_start_stop[0]),(xbox_left+win_draw+x_start_stop[0],ytop_draw+win_draw+y_start_stop[0])])
                 
    return box_list
    
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        #print(nonzero)
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def process_frame(img):
    rectangles = []
    
    scale = 1.0
    y_start_stop = [350, 700] 
    x_start_stop = [None, None] 
    rectangles.append(find_cars(img, x_start_stop, y_start_stop, scale ,color_space, svc, 
                                X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins ,show_all_rectangles=False))


    scale = 1.5
    y_start_stop = [400, None] 
    x_start_stop = [None, None]
    rectangles.append(find_cars(img, x_start_stop, y_start_stop, scale ,color_space, svc, 
                                X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,show_all_rectangles=False))
    
    scale = 2.0
    y_start_stop = [450, None] 
    x_start_stop = [None, None]    
    rectangles.append(find_cars(img, x_start_stop, y_start_stop, scale ,color_space, svc, 
                                X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,show_all_rectangles=False))    

    scale = 3.0
    y_start_stop = [550, None] 
    x_start_stop = [None, None]    
    rectangles.append(find_cars(img, x_start_stop, y_start_stop, scale ,color_space, svc, 
                                X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,show_all_rectangles=False))
    
    rectangles = [item for sublist in rectangles for item in sublist] 
    if len(rectangles) > 0:
        det.add_rects(rectangles)
    #test_img_rects = draw_boxes(img, rectangles, color=(0, 0, 255), thick=2)
    heatmap_img = np.zeros_like(img[:,:,0])
    for rect_set in det.prev_rects:
        heatmap_img = add_heat(heatmap_img, rect_set)
    #heatmap_img = add_heat(heatmap_img, rectangles)
    #heatmap_img = apply_threshold(heatmap_img, 1)
    heatmap_img = apply_threshold(heatmap_img, 1 + len(det.prev_rects)//2)    
    labels = label(heatmap_img)
    draw_img = draw_labeled_bboxes(img, labels)
    
    return draw_img

def process_video():
    
    output = 'output_videos/output.mp4'
    clip2 = VideoFileClip('project_video.mp4')#.subclip(20,25)
    clip = clip2.fl_image(process_frame)
    clip.write_videofile(output, audio=False)
    clip.reader.close()
    clip.audio.reader.close_proc()

def test_image():
    
    
    img = mpimg.imread('./test_images/test6.jpg')
    rectangles = []
    
    scale = 1.0
    y_start_stop = [350, 700] 
    x_start_stop = [None, None] 
    rectangles.append(find_cars(img, x_start_stop, y_start_stop, scale ,color_space, svc, 
                                X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins ,show_all_rectangles=False))

    scale = 1.5
    y_start_stop = [400, None] 
    x_start_stop = [None, None]
    rectangles.append(find_cars(img, x_start_stop, y_start_stop, scale ,color_space, svc, 
                                X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,show_all_rectangles=False))
    
    scale = 2.0
    y_start_stop = [450, None] 
    x_start_stop = [None, None]    
    rectangles.append(find_cars(img, x_start_stop, y_start_stop, scale ,color_space, svc, 
                                X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,show_all_rectangles=False))    

    scale = 2.5
    y_start_stop = [500, None] 
    x_start_stop = [None, None]    
    rectangles.append(find_cars(img, x_start_stop, y_start_stop, scale ,color_space, svc, 
                                X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,show_all_rectangles=False))
    
    # apparently this is the best way to flatten a list of lists
    rectangles = [item for sublist in rectangles for item in sublist] 
    test_img_rects = draw_boxes(img, rectangles, color=(0, 0, 255), thick=2)
    plt.figure(figsize=(10,10))
    plt.imshow(test_img_rects)
    
    heatmap_img = np.zeros_like(img[:,:,0])
    heatmap_img = add_heat(heatmap_img, rectangles)
    heatmap_img = apply_threshold(heatmap_img, 1)
    plt.figure(figsize=(10,10))
    plt.imshow(heatmap_img, cmap='hot')
    
    labels = label(heatmap_img)
    plt.figure(figsize=(10,10))
    plt.imshow(labels[0], cmap='gray')
    
    draw_img = draw_labeled_bboxes(img, labels)
    plt.figure(figsize=(10,10))
    plt.imshow(draw_img)
    
def test():
    image = mpimg.imread('./test_images/test1.jpg')
    draw_image = np.copy(image)
    image = image.astype(np.float32)*255  
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[350, None], 
                        xy_window=(96, 96), xy_overlap=(0.5, 0.5))
    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)                       
    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    
    plt.imshow(window_img)
    
class Vehicle_Detect():
    def __init__(self):
        # history of rectangles previous n frames
        self.prev_rects = [] 
        
    def add_rects(self, rects):
        self.prev_rects.append(rects)
        if len(self.prev_rects) > 15:
            # throw out oldest rectangle set(s)
            self.prev_rects = self.prev_rects[len(self.prev_rects)-15:]


# Load the classifier and parameters
data_file = 'ClassifierData.p'
with open(data_file, mode='rb') as f:
    data = pickle.load(f)
    
svc = data['svc'] 
X_scaler = data['X_scaler']
color_space = data['color_space']
spatial_size = data['spatial_size']
hist_bins = data['hist_bins']
orient = data['orient']
pix_per_cell = data['pix_per_cell']
cell_per_block = data ['cell_per_block']
hog_channel = data['hog_channel']
spatial_feat = data ['spatial_feat']
hist_feat = data['hist_feat']
hog_feat = data['hog_feat']

test_image()
det = Vehicle_Detect()
#process_video()