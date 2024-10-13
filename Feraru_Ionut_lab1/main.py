import cv2
import numpy as np

image = cv2.imread("lena.tif")
# print(image)
print("Size of the image: ", image.shape)
print("Length of kernel: ", len(image))

cv2.imshow('Original', image)
cv2.waitKey()
cv2.destroyAllWindows()


# blurr

blurring = np.ones((11, 11), np.float32) / 121
img = cv2.filter2D(src=image, ddepth=-1, kernel=blurring)

cv2.imshow('Original', image)
cv2.imshow('Blurred', img)

cv2.waitKey()
cv2.imwrite('lena_blured.jpg', img)
cv2.destroyAllWindows()

# blur with cv2 function
img_blur = cv2.blur(src=image, ksize=(5, 5))

cv2.imshow('Original', image)
cv2.imshow('Blurred_with_function', img_blur)

cv2.waitKey()
cv2.imwrite('blurred_with_function.jpg', img_blur)
cv2.destroyAllWindows()


# Apply Gaussian blur
# sigmaX is Gaussian Kernel standard deviation
# ksize is kernel size
gaussian_blur = cv2.GaussianBlur(src=image, ksize=(5, 5), sigmaX = 0, sigmaY = 0)

cv2.imshow('Original', image)
cv2.imshow('Gaussian Blurred', gaussian_blur)

cv2.waitKey()
cv2.imwrite('gaussian_blur.jpg', gaussian_blur)
cv2.destroyAllWindows()

# sharpened 1
sharpen_1 = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])
sharp_img = cv2.filter2D(src=image, ddepth=-1, kernel=sharpen_1)

cv2.imshow('Original', image)
cv2.imshow('Sharpened_1', sharp_img)

cv2.waitKey()
cv2.imwrite('sharp_image_1.jpg', sharp_img)
cv2.destroyAllWindows()

# sharpened 2
sharpen_2 = np.array([[0, -1, 0],
                    [-1, 6, -1],
                    [0, -1, 0]])
sharp2_img = cv2.filter2D(src=image, ddepth=-1, kernel=sharpen_2)

cv2.imshow('Original', image)
cv2.imshow('Sharpened_2', sharp2_img)

cv2.waitKey()
cv2.imwrite('sharp_image_2.jpg', sharp2_img)
cv2.destroyAllWindows()

# 4. Apply the filter
filter_4 = np.array([[0, -2, 0],
                    [-2, 8, -2],
                    [0, -2, 0]])
filter2_img = cv2.filter2D(src=image, ddepth=-1, kernel=filter_4)

cv2.imshow('Original', image)
cv2.imshow('filter_4', filter2_img)

cv2.waitKey()
cv2.imwrite('filter_4.jpg', filter2_img)
cv2.destroyAllWindows()

# rotation
height, width = image.shape[:2]
center = (width / 2, height / 2)


def rotate_at_giving_angle(angle, name_of_the_file):
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
    rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))

    cv2.imshow('Original image', image)
    cv2.imshow(name_of_the_file, rotated_image)
    # wait indefinitely, press any key on keyboard to exit
    cv2.waitKey(0)
    # write the output, the rotated image to disk
    cv2.imwrite(name_of_the_file + '.jpg', rotated_image)


rotate_at_giving_angle(45, "rotated_at_45")
rotate_at_giving_angle(82, "rotated_at_82")
rotate_at_giving_angle(-82, "rotated_at_-82")
rotate_at_giving_angle(-180, "rotated_at_-180")

# [start_row:end_row, start_col:end_col]
image_size = image.size

def crop_image(upper_left_pixel, width, length):
    end_row = upper_left_pixel + length
    end_col = upper_left_pixel + width

    cropped_image = image[upper_left_pixel:end_row, upper_left_pixel:end_col]

    # Display cropped image
    cv2.imshow("cropped", cropped_image)

    # Save the cropped image
    cv2.imwrite("Cropped Image.jpg", cropped_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


crop_image(0, 500, 200)
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
