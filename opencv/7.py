import cv2

faceCascade=cv2.CascadeClassifier("resources/face_detection.xml")

cap=cv2.VideoCapture(0)

while True:
    success,frame=cap.read()

    faces=faceCascade.detectMultiScale(frame,1.1,4)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(56,223,0),2)
    
    cv2.imshow('Output',frame)

    if cv2.waitKey(1)==ord('q'):
        break