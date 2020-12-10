import numpy as np
import cv2 as cv

# From opencv
cap = cv.VideoCapture("1_2016-05-02_23-06-32.mp4")
# read the first frame of the video
ret, frame1 = cap.read()
# below is used for cropping the video
# crop = 120
# frame1 = frame1[crop:, :]

prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
# create an array of equal size to the frame
hsv = np.zeros_like(frame1)
# set all pixels to black
hsv[...,1] = 255
while(1):
    ret, frame2 = cap.read()
    # for cropping --> frame2 = frame2[crop:, :]
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    # utilizing Gunner Farneback's algorithm -- need to read more about how this works
    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # direction corresponds to Hue value of image, magnitude corresponds to Value plane
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    # the following method of thresholding is pretty inefficient
    thresh = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    y = []
    for i in thresh:
        y.append([j if j > 120 else 0 for j in i ])
    hsv[...,2] = y
    # hsv[...,2] = thresh
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    cv.imshow('frame2 original', frame2)
    cv.imshow('frame2',bgr)
    # 30 ms delay and looks only at the last 8 bits
    k = cv.waitKey(10) & 0xff
    if k == 27:
        break
    # when pressing s key, will save the photo
    elif k == ord('s'):
        cv.imwrite('opticalfb5.png',frame2)
        cv.imwrite('opticalhsv5.png',bgr)
    prvs = next

