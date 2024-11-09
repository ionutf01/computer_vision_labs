import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

figure = plt.figure(figsize=(13,10))
rows = 3
columns = 4


def plot_image(image, title, place_of_the_image):
    figure.add_subplot(rows, columns, place_of_the_image)
    plt.imshow(image)
    plt.axis('on')
    plt.title(title)

def simple_averaging_grayscale(img):
    (row, col) = img.shape[0:2]
    for i in range(row):
        for j in range(col):
            # simple averaging
            img[i, j] = sum(img[i, j] / 3)
    plot_image(img, "simple_averaging_grayscale", 1)

def weighted_average_grayscale_1(img):
    (row, col) = img.shape[0:2]
    for i in range(row):
        for j in range(col):
            # weighted average
            img[i, j] = img[i, j][2]*0.11+img[i,j][1]*0.59+img[i, j][0]*.3
    plot_image(img, "weighted_average_grayscale", 2)

def weighted_average_grayscale_2(img):
    (row, col) = img.shape[0:2]
    for i in range(row):
        for j in range(col):
            # weighted average
            img[i, j] = img[i, j][0]*0.0722+img[i,j][1]*0.7152+img[i, j][2]*0.2126
    plot_image(img, "weighted_average_grayscale_2", 3)

def weighted_average_grayscale_3(img):
    (row, col) = img.shape[0:2]
    for i in range(row):
        for j in range(col):
            # weighted average
            img[i, j] = img[i, j][0]*0.114+img[i,j][1]*0.587+img[i, j][2]*0.299
    plot_image(img, "weighted_average_grayscale_3", 4)

def desaturation(img):
    (row, col) = img.shape[0:2]
    for i in range(row):
        for j in range(col):
            img[i, j] = (min(img[i, j]) + max(img[i, j]))/2

    plot_image(img, "desaturation", 5)
def decomposition_max_gray(img):
    (row, col) = img.shape[0:2]
    for i in range(row):
        for j in range(col):
            # weighted average
            img[i, j] = max(img[i, j])

    plot_image(img, "desaturation_max_gray", 6)
def decomposition_min_gray(img):
    (row, col) = img.shape[0:2]
    for i in range(row):
        for j in range(col):
            # weighted average
            img[i, j] = min(img[i, j])

    plot_image(img, "decomposition_min_gray", 7)

# Calling all the functions
img = cv2.imread('lena.tif')
simple_averaging_grayscale(img)
img = cv2.imread('lena.tif')
weighted_average_grayscale_1(img)
img = cv2.imread('lena.tif')
weighted_average_grayscale_2(img)
img = cv2.imread('lena.tif')
weighted_average_grayscale_3(img)
img = cv2.imread('lena.tif')
desaturation(img)
img = cv2.imread('lena.tif')
decomposition_max_gray(img)
img = cv2.imread('lena.tif')
decomposition_min_gray(img)
img = cv2.imread('lena.tif')
(B, G, R) = cv2.split(img)
zeros = np.zeros(img.shape[:2], dtype="uint8")
print("ZEROS : ", zeros)
red = cv2.merge([zeros, zeros, R])
green= cv2.merge([zeros, G, zeros])
blue= cv2.merge([B, zeros, zeros])
plot_image(red, "red only", 8)
plot_image(green, "green only", 9)
plot_image(blue, "blue only", 10)


def custom_number_of_gray_shades(img, num_shades):
    (row, col) = img.shape[0:2]
    for i in range(row):
        for j in range(col):
            img[i, j] = sum(img[i, j] / 3)
    print("NUM OF SHADES: ", num_shades)

    p = max(1, min(255, num_shades))
    # interval = 256 // p
    interval = sum(range(0, p)) // p

    lookup_table = np.zeros(256, np.uint16)

    for i in range(p):
        start = i * interval
        end = start + interval
        if i == p - 1:
            end = 256
        avg_value = (start + end - 1) // 2
        lookup_table[start:end] = avg_value
    custom_grey_image = lookup_table[img]
    plot_image(custom_grey_image, "custom_number_of_gray_shades", 11)

def floyd_steinberg_dithering(img):
    dithered_image = img.copy().astype(np.float32)
    (row, col) = img.shape[0:2]
    for i in range(row):
        for j in range(col):
            old_pixel = dithered_image[i, j]
            new_pixel = 255 if old_pixel > 127 else 0
            dithered_image[i, j] = new_pixel
            quant_error = old_pixel - new_pixel

            if j + 1 < col:
                dithered_image[i, j+1] += quant_error * 7/16
            if i + 1 < row:
                if j > 0:
                    dithered_image[i + 1][j - 1] += quant_error * 3/16
                dithered_image[i + 1, j] += quant_error * 5/16
                if j + 1 < col:
                    dithered_image[i+1, j+1] += quant_error * 1/16
    cv2.imshow("Original image", img)
    cv2.imshow("Floyd Steinberg Dithering algorithm applied", dithered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # plot_image(img, "floyd_steinberg", 12)

def stucki_dithering(img):
    dithered_image = img.copy().astype(np.float32)
    (row, col) = img.shape[0:2]

    coefficients = [
        [0, 0, 0, 8, 4],
        [2, 4, 8, 4, 2],
        [1, 2, 4, 2, 1]
    ]
    divisor = 42


    for i in range(row):
        for j in range(col):
            old_pixel = dithered_image[i, j]
            new_pixel = 255 if old_pixel > 127 else 0
            dithered_image[i, j] = new_pixel
            quant_error = old_pixel - new_pixel

            for di in range (-1, 2):
                for dj in range (-2, 3):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < col:
                        dithered_image[ni, nj] += quant_error * coefficients[di + 1][dj + 2] / divisor
    cv2.imshow("Original image", img)
    cv2.imshow("Stucki Dithering algorithm applied", dithered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('lena.tif')
num_shades = random.randint(1, 255)
custom_number_of_gray_shades(img, num_shades)

img = cv2.imread('lena.tif', cv2.IMREAD_GRAYSCALE)
floyd_steinberg_dithering(img)

img = cv2.imread('lena.tif', cv2.IMREAD_GRAYSCALE)
stucki_dithering(img)

def grayscale_to_rgb(img):
    colored_image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    (row, col) = colored_image.shape[0:2]
    for i in range(row):
        for j in range(col):
            if colored_image[i,j][2] < 100:
                colored_image[i,j] = (200,100,200)
    plot_image(colored_image, "grayscale_to_rgb", 12)

img = cv2.imread('lena.tif', cv2.IMREAD_GRAYSCALE)
grayscale_to_rgb(img)
plt.show()

