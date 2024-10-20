import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
figure = plt.figure(figsize=(13,10))
rows = 3
columns = 4

image = mpimg.imread("lena.tif")
print("Size of the image: ", image.shape)
print("Length of kernel: ", len(image))

def plot_image(image, title, place_of_the_image):
    figure.add_subplot(rows, columns, place_of_the_image)
    plt.imshow(image)
    plt.axis('on')
    plt.title(title)

plot_image(image, "Original", 1)

blurring = np.ones((8, 8), np.float32) / 64
img = cv2.filter2D(src=image, ddepth=-1, kernel=blurring)
plot_image(img, "blurr_1", 2)

# blur with cv2 function
img_blur = cv2.blur(src=image, ksize=(5, 5))
plot_image(img_blur, "blurr_2", 3)

# Apply Gaussian blur
# sigmaX is Gaussian Kernel standard deviation
# ksize is kernel size
gaussian_blur = cv2.GaussianBlur(src=image, ksize=(5, 5), sigmaX = 0, sigmaY = 0)
plot_image(gaussian_blur, "blurr_3_gaussian", 4)

# sharpened 1
sharpen_1 = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])
sharp_img = cv2.filter2D(src=image, ddepth=-1, kernel=sharpen_1)
plot_image(sharp_img, "sharp_1", 5)

sharpen_2 = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])
sharp2_img = cv2.filter2D(src=image, ddepth=-1, kernel=sharpen_2)
plot_image(sharp2_img, "sharp_2", 6)

# 4. Apply the filter
filter_4 = np.array([[0, -2, 0],
                    [-2, 8, -2],
                    [0, -2, 0]])
filter2_img = cv2.filter2D(src=image, ddepth=-1, kernel=filter_4)
plot_image(filter2_img, "given_filter_applied", 7)

# rotation
height, width = image.shape[:2]
center = (width / 2, height / 2)

def rotate_at_giving_angle(angle, position):
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
    rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))

    plot_image(rotated_image, "Rotated at angle: "+str(angle), position)


rotate_at_giving_angle(45, 8)
rotate_at_giving_angle(82, 9)
rotate_at_giving_angle(-82, 10)
rotate_at_giving_angle(-180, 11)

# [start_row:end_row, start_col:end_col]
image_size = image.size

def crop_image(upper_left_pixel, width, length):
    if upper_left_pixel[0] > 0 and upper_left_pixel[1] > 0 and width < image.shape[0] and length < image.shape[1]:
        end_row = upper_left_pixel[0] + length
        end_col = upper_left_pixel[1] + width
        cropped_image = image[upper_left_pixel[0]:end_row, upper_left_pixel[1]:end_col]
        plot_image(cropped_image, "cropped_image", 12)
    else:
        print('crop is not possible, width or length are bigger than the image_size; image_size: ', image_size)


crop_image((100,100), 100,500)


def draw_emoji():
    white = (255, 255, 255)
    black = (0, 0, 0)
    yellow = (100, 255, 255)
    blue = (255, 100, 0)
    canvas = np.zeros((300, 300, 3), dtype="uint8")
    canvas.fill(255)
    (centerX, centerY) = (canvas.shape[1] // 2, canvas.shape[0] // 2)
    # big yellow circle
    cv2.circle(canvas, (centerX, centerY), 100, yellow, thickness=-1)
    # black eyes
    cv2.circle(canvas, (100, 120), 25, black, thickness=-1)
    cv2.circle(canvas, (200, 120), 25, black, thickness=-1)
    # white effect on eyes
    cv2.circle(canvas, (195, 112), 13, white, thickness=-1)
    cv2.circle(canvas, (95, 112), 13, white, thickness=-1)

    cv2.circle(canvas, (205, 130), 5, white, thickness=-1)
    cv2.circle(canvas, (105, 130), 5, white, thickness=-1)
    # smile
    cv2.ellipse(canvas, (150, 170), (50, 50), 0, 0, 180, black, -1)
    # tears
    cv2.ellipse(canvas, (100, 135), (25, 10), 0, 0, 210, blue, -1)
    cv2.ellipse(canvas, (200, 135), (25, 10), 0, 0, 210, blue, -1)

    cv2.imshow("Canvas", canvas)
    cv2.imwrite('Feraru_Ionut_MSD1_lab1_emoji.jpg',canvas )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

draw_emoji()
plt.show()