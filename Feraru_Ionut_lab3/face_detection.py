import cv2
import numpy as np


def detect_skin_pixels_3(image, position):
    skin_image = np.zeros_like(image)
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    (row, col) = image.shape[0:2]
    for i in range(row):
        for j in range(col):
            y = ycbcr_image[i, j][0]
            cr = ycbcr_image[i, j][1]
            cb = ycbcr_image[i, j][2]
            if 80 <= y <= 255 and 180 >= cr >= 135 >= cb >= 85:
                skin_image[i, j] = (255, 255, 255)
            else:
                skin_image[i, j] = (0, 0, 0)

    return skin_image

# Detect faces using the YCbCr skin detection method
image = cv2.imread('Pratheepan_Dataset/FacePhoto/m(01-32)_gr.jpg')
cv2.imshow('Original', image)

skin_image = detect_skin_pixels_3(image, 1)
cv2.imshow('skin_image', skin_image)

gray_skin_image = cv2.cvtColor(skin_image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray_skin_image', gray_skin_image)

binary_skin_image = cv2.inRange(gray_skin_image, 127, 255)
cv2.imshow('binary_skin_image', binary_skin_image)

contours, _ = cv2.findContours(binary_skin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Detected Face', image)
cv2.waitKey(0)
cv2.destroyAllWindows()