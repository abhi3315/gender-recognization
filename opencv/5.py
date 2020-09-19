import cv2
import numpy as np

img=np.zeros((512,512,3),np.uint8)

cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(0,255,0),3)
cv2.rectangle(img,(0,0),(300,250),(135,66,0),4)
cv2.circle(img,(300,100),50,(243,66,98),-1)
cv2.putText(img,"Opencv",(300,200),cv2.FONT_HERSHEY_DUPLEX,1,(0,150,0),2)

cv2.imshow('Output',img)

cv2.waitKey(0)