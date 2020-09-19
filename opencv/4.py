import cv2

img=cv2.imread('resources/lambo.PNG')
print(img.shape)

imgResize=cv2.resize(img,(300,200))
imgCropped=img[0:200,200:500]
print(imgResize.shape)

cv2.imshow('Img',img)
cv2.imshow('Resize',imgResize)
cv2.imshow('Cropped',imgCropped)

cv2.waitKey(0)