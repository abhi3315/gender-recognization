import cv2
import numpy as np

kernel=np.ones((5,5),np.uint8)

img=cv2.imread('resources/lena.png')
grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur=cv2.GaussianBlur(img,(9,9),10)
imgCanny=cv2.Canny(img,100,100)
imgDialation=cv2.dilate(img,kernel,iterations=10)
imgErode=cv2.erode(img,kernel,iterations=1)

cv2.imshow('Img',img)
cv2.imshow('Gray Img',grayImg)
cv2.imshow('Blur Img',imgBlur)
cv2.imshow('Canny Img',imgCanny)
cv2.imshow('Dilated Img',imgDialation)
cv2.imshow('Eroded Img',imgErode)

if cv2.waitKey(0)==ord('q'):
    cv2.destroyAllWindows()