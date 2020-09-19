import cv2

faceCascade=cv2.CascadeClassifier("resources/face_detection.xml")

img=cv2.imread('resources/lena.png')
faces=faceCascade.detectMultiScale(img,1.1,4)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(56,223,0),2)

cv2.imshow('Result',img)

cv2.waitKey(0)