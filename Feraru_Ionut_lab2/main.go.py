import cv2

def simple_averaging_grayscale(img):
    (row, col) = img.shape[0:2]
    for i in range(row):
        for j in range(col):
            # simple averaging
            img[i, j] = sum(img[i, j] / 3)

    # cv2.imshow('simple_averaging_grayscal', img)
    cv2.imwrite("average_grayscale.png", img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()


def weighted_average_grayscale_1(img):
    (row, col) = img.shape[0:2]
    for i in range(row):
        for j in range(col):
            # weighted average
            img[i, j] = img[i, j][0]*0.11+img[i,j][1]*0.59+img[i, j][2]*0.3

    # cv2.imshow('weighted_average_grayscale_1', img)
    cv2.imwrite("weighted_grayscale_1.png", img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

def weighted_average_grayscale_2(img):
    (row, col) = img.shape[0:2]
    for i in range(row):
        for j in range(col):
            # weighted average
            img[i, j] = img[i, j][0]*0.0722+img[i,j][1]*0.7152+img[i, j][2]*0.2126

    # cv2.imshow('weighted_average_grayscale_2', img)
    cv2.imwrite("weighted_grayscale_2.png", img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

def weighted_average_grayscale_3(img):
    (row, col) = img.shape[0:2]
    for i in range(row):
        for j in range(col):
            # weighted average
            img[i, j] = img[i, j][0]*0.114+img[i,j][1]*0.587+img[i, j][2]*0.299

    # cv2.imshow('weighted_average_grayscale_3', img)
    cv2.imwrite("weighted_grayscale_3.png", img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
def desaturation(img):
    (row, col) = img.shape[0:2]
    for i in range(row):
        for j in range(col):
            # weighted average
            img[i, j] = (min(img[i, j]) + max(img[i, j]))/2

    # cv2.imshow('desaturation', img)
    cv2.imwrite("desaturation.png", img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
def decomposition_max_gray(img):
    (row, col) = img.shape[0:2]
    for i in range(row):
        for j in range(col):
            # weighted average
            img[i, j] = max(img[i, j])

    # cv2.imshow('decomposition_max_grey', img)
    cv2.imwrite("decomposition_max_gray.png", img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
def decomposition_min_gray(img):
    (row, col) = img.shape[0:2]
    for i in range(row):
        for j in range(col):
            # weighted average
            img[i, j] = min(img[i, j])

    # cv2.imshow('decomposition_min_grey', img)
    cv2.imwrite("decomposition_min_gray.png", img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

def single_colour_channel(img):
    cv2.imshow("B channel", img[:,:,2])
    cv2.imshow("G channel", img[:,:,1])
    cv2.imshow("R channel", img[:,:,0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()



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
single_colour_channel(img)

def custom_number_of_gray_shades():
    (row, col) = img.shape[0:2]
    for i in range(row):
        for j in range(col):
            # simple averaging
            img[i, j] = sum(img[i, j] / 3)
    