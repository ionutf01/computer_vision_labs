from PIL import Image
import pytesseract
import argparse
import cv2
import os
import numpy as np
from skimage.util import random_noise

def salt_pepper_noise(image):
    # Add salt-and-pepper noise to the image.
    noise_img = random_noise(image, mode='s&p', amount=0.1)

    # The above function returns a floating-point image
    # on the range [0, 1], thus we changed it to 'uint8'
    # and from [0,255]
    noise_img = np.array(255 * noise_img, dtype='uint8')
    return noise_img

def add_gaussian_noise(image):
    # Generate Gaussian noise
    gauss = np.random.normal(0, 1, image.size)
    gauss = gauss.reshape(image.shape[0], image.shape[1]).astype('uint8')
    # Add the Gaussian noise to the image
    img_gauss = cv2.add(image, gauss)
    return img_gauss

def add_speckle_noise(image):
    gauss = np.random.normal(0, 1, image.size)
    gauss = gauss.reshape(image.shape[0], image.shape[1]).astype('uint8')
    noise = image + image * gauss
    return noise

def apply_affine_rotation_with_custom_angle(image):
    # Rotate the image by 45 degrees
    (rows, cols) = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))
    return rotated_image

def apply_affine_vertical_shear(image):
    # Shear the image vertically
    (rows, cols) = image.shape[:2]
    M = np.float32([[1, 0.5, 0], [0, 1, 0]])
    sheared_image = cv2.warpAffine(image, M, (cols, rows))
    return sheared_image

def apply_affine_horizontal_shear(image):
    # Shear the image horizontally
    (rows, cols) = image.shape[:2]
    M = np.float32([[1, 0, 0], [0.5, 1, 0]])
    sheared_image = cv2.warpAffine(image, M, (cols, rows))
    return sheared_image

def resize_by_keeping_aspect_ratio(image, width=None, height=None):
    # Resize the image by keeping the aspect ratio
    aspect_ratio = width / height
    (h, w) = image.shape[:2]
    if width is None:
        new_width = int(h / aspect_ratio)
        new_height = height
    elif height is None:
        new_width = width
        new_height = int(w * aspect_ratio)
    else:
        new_width = width
        new_height = height
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

def resize_by_distorting_aspect_ratio(image, width=None, height=None):
    # Resize the image by distorting the aspect ratio
    resized_image = cv2.resize(image, (width, height))
    return resized_image

def blur_image_1(image):
    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred_image
def blur_image_2(image):
    # Apply median blur to the image
    blurred_image = cv2.medianBlur(image, 5)
    return blurred_image
def blur_image_3(image):
    # Apply bilateral filter to the image
    blurred_image = cv2.bilateralFilter(image, 9, 75, 75)
    return blurred_image

def sharpen_image(image):
    # Apply sharpening to the image
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

def apply_erosion(image):
    # Apply erosion to the image
    kernel = np.ones((5, 5), np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=1)
    return eroded_image

def apply_dilation(image):
    # Apply dilation to the image
    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    return dilated_image

def apply_opening(image):
    # Apply opening to the image
    kernel = np.ones((5, 5), np.uint8)
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opened_image

def apply_closing(image):
    # Apply closing to the image
    kernel = np.ones((5, 5), np.uint8)
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closed_image
def apply_morphological_gradient(image):
    # Apply morphological gradient to the image
    kernel = np.ones((5, 5), np.uint8)
    gradient_image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    return gradient_image


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image to be OCR'd")
ap.add_argument("-p", "--preprocess", type=str, default="thresh",
    help="type of preprocessing to be done")
args = vars(ap.parse_args())

# load the example image and convert it to grayscale
image = cv2.imread(args["image"])
# noisy_image = add_gaussian_noise(image)
# # noisy_image = add_speckle_noise(image)
# noisy_image = salt_pepper_noise(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# check to see if we should apply thresholding to preprocess the
# image
if args["preprocess"] == "thresh":
    gray = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# make a check to see if median blurring should be done to remove
# noise
elif args["preprocess"] == "blur":
    gray = cv2.medianBlur(gray, 3)
# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

# load the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary file
text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(text)
ground_truth = "The quick brown fox jumps over the lazy dog"

def compare_text_with_ground_truth_remove_spaces(text, ground_truth):
    text = text.replace(" ", "")
    ground_truth = ground_truth.replace(" ", "")
    if text == ground_truth:
        return True
    else:
        return False

def apply_each_noise_method_and_compare_the_text_with_ground_truth():
    # Apply each noise method and compare the text with the ground truth
    # 1. Salt and Pepper Noise
    noisy_image = salt_pepper_noise(gray)
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, noisy_image)
    noisy_text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    with open("output_salt_pepper.txt", "w") as file:
        file.write("Salt and Pepper Noise: " + noisy_text + "\n")
        file.write("Matched: " + str(compare_text_with_ground_truth_remove_spaces(noisy_text, ground_truth)) + "\n")
    # 2. Gaussian Noise
    noisy_image = add_gaussian_noise(gray)
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, noisy_image)
    noisy_text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    with open("output_gaussian.txt", "w") as file:
        file.write("Gaussian Noise: " + noisy_text + "\n")
        file.write("Matched: " + str(compare_text_with_ground_truth_remove_spaces(noisy_text, ground_truth)) + "\n")
    # 3. Speckle Noise
    noisy_image = add_speckle_noise(gray)
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, noisy_image)
    noisy_text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    with open("output_speckle.txt", "w") as file:
        file.write("Speckle Noise: " + noisy_text + "\n")
        file.write("Matched: " + str(compare_text_with_ground_truth_remove_spaces(noisy_text, ground_truth)) + "\n")
	# 4. Affine Rotation
	rotated_image = apply_affine_rotation_with_custom_angle(gray)
	filename = "{}.png".format(os.getpid())
	cv2.imwrite(filename, rotated_image)
	rotated_text = pytesseract.image_to_string(Image.open(filename))
	os.remove(filename)
	with open("output_rotation.txt", "w") as file:
		file.write("Affine Rotation: " + rotated_text + "\n")
		file.write("Matched: " + str(compare_text_with_ground_truth_remove_spaces(rotated_text, ground_truth)) + "\n")
	# 5. Affine Vertical Shear
	sheared_image = apply_affine_vertical_shear(gray)
	filename = "{}.png".format(os.getpid())
	cv2.imwrite(filename, sheared_image)
	sheared_text = pytesseract.image_to_string(Image.open(filename))
	os.remove(filename)
	with open("output_vertical_shear.txt", "w") as file:
		file.write("Affine Vertical Shear: " + sheared_text + "\n")
		file.write("Matched: " + str(compare_text_with_ground_truth_remove_spaces(sheared_text, ground_truth)) + "\n")
	# 6. Affine Horizontal Shear
	sheared_image = apply_affine_horizontal_shear(gray)
	filename = "{}.png".format(os.getpid())
	cv2.imwrite(filename, sheared_image)
	sheared_text = pytesseract.image_to_string(Image.open(filename))
	os.remove(filename)
	with open("output_horizontal_shear.txt", "w") as file:
		file.write("Affine Horizontal Shear: " + sheared_text + "\n")
		file.write("Matched: " + str(compare_text_with_ground_truth_remove_spaces(sheared_text, ground_truth)) + "\n")
	# 7. Resize by Keeping Aspect Ratio
	resized_image = resize_by_keeping_aspect_ratio(gray, width=500)
	filename = "{}.png".format(os.getpid())
	cv2.imwrite(filename, resized_image)
	resized_text = pytesseract.image_to_string(Image.open(filename))
	os.remove(filename)
	with open("output_resize_aspect_ratio.txt", "w") as file:
		file.write("Resize by Keeping Aspect Ratio: " + resized_text + "\n")
		file.write("Matched: " + str(compare_text_with_ground_truth_remove_spaces(resized_text, ground_truth)) + "\n")
	# 5. Resize by Distorting Aspect Ratio
	resized_image = resize_by_distorting_aspect_ratio(gray, width=500, height=500)
	filename = "{}.png".format(os.getpid())
	cv2.imwrite(filename, resized_image)
	resized_text = pytesseract.image_to_string(Image.open(filename))

# show the output images

apply_each_noise_method_and_compare_the_text_with_ground_truth()


# cv2.imshow("Image", i)
cv2.imshow("Output", gray)
cv2.waitKey(0)

