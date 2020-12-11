import numpy as np
import cv2 as cv

# From opencv
cap = cv.VideoCapture("1_2016-05-02_22-46-25.mp4")
# read the first frame of the video
ret, frame1 = cap.read()
# below is used for cropping the video
# crop = 120
# frame1 = frame1[crop:, :]

prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
cv.imshow('frame2 original', prvs)
# create an array of equal size to the frame
hsv = np.zeros_like(frame1)
# set all pixels to black
hsv[...,1] = 255


while(1):
    ret, frame2 = cap.read()
    # for cropping --> frame2 = frame2[crop:, :]
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    next = cv.fastNlMeansDenoising(next, None, 15, 7, 21)
    cv.imshow('test', next)

    # utilizing Gunner Farneback's algorithm -- need to read more about how this works
    flow = cv.calcOpticalFlowFarneback(prev=prvs, next=next, flow=None, pyr_scale=0.5, levels=3, winsize=30, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    # direction corresponds to Hue value of image, magnitude corresponds to Value plane
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    # the following method of thresholding is pretty inefficient
    thresh = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    #a = cv.adaptiveThreshold(np.uint8(thresh), 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    ret, y = cv.threshold(thresh,110,255,cv.THRESH_TOZERO)
    hsv[...,2] = y
    #print(np.array(y))
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

