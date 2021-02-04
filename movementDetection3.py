#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 15:38:37 2021

@author: jamisonmeindl
"""

from __future__ import print_function
import cv2 as cv
import numpy as np 
from numpy import load

color=(255,0,0)
thickness=2


kernel = np.ones((3,3),np.uint8)
kernel2 = np.ones((1,10), np.uint8)
kernel3 = np.ones((3,1),np.uint8)


testC = []
selected = False

lastPoint = []
counter = 0
wait = 0
fishShape = load('fishImage.npy')


backSub = cv.createBackgroundSubtractorMOG2()
#backSub = cv.createBackgroundSubtractorKNN()
capture = cv.VideoCapture('testImages/combinedVideo.mov')

frame_width = int(capture.get(3)) 
frame_height = int(capture.get(4)) 
   
size = (frame_width, frame_height)

'''
result = cv.VideoWriter('sampleBoxing.avi',  
                         cv.VideoWriter_fourcc(*'MJPG'), 
                         20, size) 
'''

while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    fgMask = backSub.apply(frame)
    
    
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    cv.rectangle(frame, (495, 2), (605,20), (255,255,255), -1)
    cv.putText(frame, 'Counter: ' + str(counter), (500, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    #blur = cv2.GaussianBlur(gray,(21,21),0)
    ret,thresh = cv.threshold(fgMask,50,255,cv.THRESH_BINARY)
    
    
    fgMask = cv.erode(thresh, kernel, iterations = 3)
    fgMask = cv.dilate(fgMask, kernel2, iterations = 2)
    fgMask = cv.dilate(fgMask, kernel3, iterations = 2)
    
    
    
    contours, hierarchy = cv.findContours(fgMask,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
    
    
    if len(contours) != 0:
        for c in contours:
            rect = cv.boundingRect(c)
            height, width = fgMask.shape[:2]     
            #print(rect)
            match = cv.matchShapes(fishShape, c, 1,0.0)
            
            if rect[2] > 0.075*height and rect[2] < 0.7*height and rect[3] > 0.1*width and rect[3] < 0.7*width: 
                x,y,w,h = cv.boundingRect(c)  
                
                if match < 1.5:
                    if abs(x + (w / 2) - (width / 2)) < 75:
                        if wait == 0:
                            counter += 1
                        wait = 12
                    cv.drawContours(frame, c, -1, color, thickness)
                    cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)  # draw red bounding box in img
                    
            elif rect[2] > 0.05*height and rect[2] < 0.7*height and rect[3] > 0.05*width and rect[3] < 0.7*width: 
                x,y,w,h = cv.boundingRect(c)    
                if match < 1.5:
                    if abs(x + (w / 2) - (width / 2)) < 75:
                        if wait == 0:
                            counter += 1
                        wait = 6
                    cv.drawContours(frame, c, -1, (0, 255,0), 1)
                    cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            else:
                img2 = fgMask
    else:
        img2 = fgMask
    
    #cv.drawContours(frame, fishShape, -1, color, thickness)
    cv.imshow('Boxed Video', frame)
    cv.imshow('Transformations', fgMask)
    #result.write(frame) 
    if wait > 0:
        wait -= 1
    keyboard = cv.waitKey(20)
    if keyboard == 'q' or keyboard == 27:
        break
    
#result.release()
capture.release()

cv.destroyAllWindows()
