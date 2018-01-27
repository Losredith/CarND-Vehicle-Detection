#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 18:30:53 2018

@author: laibin
"""

import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from functions import get_hog_features


def explore(cars,notcars):
    fig, axs = plt.subplots(2,2, figsize=(10, 10))
    fig.subplots_adjust(hspace = .2, wspace=.001)
    axs = axs.ravel()

    for i in np.arange(2):
        img = cv2.imread(cars[np.random.randint(0,len(cars))])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        axs[i].axis('off')
        axs[i].set_title('car', fontsize=10)
        axs[i].imshow(img)
    for i in np.arange(2,4):
        img = cv2.imread(notcars[np.random.randint(0,len(notcars))])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        axs[i].axis('off')
        axs[i].set_title('nocar', fontsize=10)
        axs[i].imshow(img)
        
def showHOGfeature(car, notcar):
    features, car_dst = get_hog_features(car[:,:,2], 9, 8, 8, vis=True, feature_vec=True)
    features, noncar_dst = get_hog_features(notcar[:,:,2], 9, 8, 8, vis=True, feature_vec=True)
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7,7))
    f.subplots_adjust(hspace = .4, wspace=.2)
    ax1.imshow(car)
    ax1.set_title('Car Image', fontsize=16)
    ax2.imshow(car_dst, cmap='gray')
    ax2.set_title('Car HOG', fontsize=16)
    ax3.imshow(notcar)
    ax3.set_title('Non-Car Image', fontsize=16)
    ax4.imshow(noncar_dst, cmap='gray')
    ax4.set_title('Non-Car HOG', fontsize=16)
        


images = glob.glob('training_data/*/*/*.png')
# Read in cars and notcars
cars = []
notcars = []

for image in images:
    if 'non-vehicles' in image :
        notcars.append(image)
    else:
        cars.append(image)
        
explore(cars,notcars)
car_img = mpimg.imread(cars[np.random.randint(0,len(cars))])
noncar_img = mpimg.imread(notcars[np.random.randint(0,len(notcars))])
showHOGfeature(car_img,noncar_img)