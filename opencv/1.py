import cv2
print('Opencv Imported')

img=cv2.imread('resources/lena.png')
cv2.imshow('Image',img)
cv2.waitKey(0)
print(img)

