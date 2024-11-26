from random import gauss
from statistics import median

from PIL import Image
import pytesseract
import argparse
import cv2
import os
import numpy as np
from skimage.util import random_noise


def compare_text_with_ground_truth_remove_spaces(text, ground_truth):
    text = ' '.join(text.split()).lower()
    ground_truth = ' '.join(ground_truth.split()).lower()

    # Convert texts to sets of words
    text_words = set(text.split())
    ground_truth_words = set(ground_truth.split())

    # Calculate intersection and union
    intersection = text_words.intersection(ground_truth_words)
    union = text_words.union(ground_truth_words)

    # Calculate the percentage of matching words
    if not union:
        return 100.0  # If both sets are empty, consider them 100% matching

    match_percentage = (len(intersection) / len(union)) * 100

    # Print debugging information
    print("COMPARE FUNCTION RESULTS: ---------------")
    print("Extracted text: ", text)
    print("Ground truth: ", ground_truth)
    print("Intersection: ", intersection)
    print("Union: ", union)
    print("Match percentage: {:.2f}%".format(match_percentage))
    print("---------------------------------")
    return match_percentage


def getGroundTruthForImage(image):
    return pytesseract.image_to_string(image)


##
# Defining Text image
##
text_image_name = "text.png"
text_image = cv2.imread(text_image_name)
text_image_gray = cv2.cvtColor(text_image, cv2.COLOR_BGR2GRAY)

##
# Defining ocr-test image
##
ocr_test_image_name = "ocr-test.png"
ocr_test_image = cv2.imread(ocr_test_image_name)
ocr_test_image_gray = cv2.cvtColor(ocr_test_image, cv2.COLOR_BGR2GRAY)

##
# Defining sample21
##
sample21_image_name = "sample21.jpg"
sample21_image = cv2.imread(sample21_image_name)
sample21_gray = cv2.cvtColor(sample21_image, cv2.COLOR_BGR2GRAY)


def salt_pepper_noise(image, amount):
    # Add salt-and-pepper noise to the image.
    noise_img = random_noise(image, mode='s&p', amount=amount)

    # The above function returns a floating-point image
    # on the range [0, 1], thus we changed it to 'uint8'
    # and from [0,255]
    noise_img = np.array(255 * noise_img, dtype='uint8')
    return noise_img


def salt_pepper_comparison(image, image_name):
    amount_for_salt_pepper = 0.01
    noisy_image = salt_pepper_noise(image, amount_for_salt_pepper)
    noisy_text = pytesseract.image_to_string(noisy_image)
    print("NOISY TEXT: ", noisy_text)
    ground_truth = getGroundTruthForImage(image)
    print("Ground truth: ", ground_truth)
    output = [
        "Image name: " + image_name + "\n",
        # "Text after salt and Pepper Noise: " + noisy_text + "\n",
        "Amount: " + str(amount_for_salt_pepper) + "\n",
        # "Ground truth: " + ground_truth + "\n",
        "Matched: " + str(compare_text_with_ground_truth_remove_spaces(noisy_text, ground_truth)) + "\n",
        '\n'
    ]
    with open("output_salt_pepper.txt", "a") as file:
        file.writelines(output)


salt_pepper_comparison(text_image, text_image_name)
salt_pepper_comparison(ocr_test_image, ocr_test_image_name)
salt_pepper_comparison(sample21_image, sample21_image_name)


def add_gaussian_noise(image, mean=0, stddev=1):
    # Generate Gaussian noise
    gauss = np.random.normal(mean, stddev, image.shape).astype('float32')
    # Add the Gaussian noise to the image
    noisy_image = image + gauss
    # Clip the pixel values to be in the proper range
    noisy_image = np.clip(noisy_image, 0, 255).astype('uint8')
    return noisy_image


def gaussian_noise_comparison(image, image_name):
    noisy_image = add_gaussian_noise(image)
    noisy_text = pytesseract.image_to_string(noisy_image)
    # print("NOISY TEXT: ", noisy_text)
    ground_truth = getGroundTruthForImage(image)
    output = [
        "Image name: " + image_name + "\n",
        # "Text after adding gaussian noise: " + noisy_text + "\n",
        # "Ground truth: " + ground_truth + "\n",
        "Matched: " + str(compare_text_with_ground_truth_remove_spaces(noisy_text, ground_truth)) + "\n",
        '\n'
    ]
    with open("output_gaussian_noise.txt", "a") as file:
        file.writelines(output)


gaussian_noise_comparison(text_image, text_image_name)
gaussian_noise_comparison(ocr_test_image, ocr_test_image_name)
gaussian_noise_comparison(sample21_image, sample21_image_name)


def add_speckle_noise(image):
    # Ensure the image is a NumPy array with float32 type for processing
    image = np.asarray(image, dtype=np.float32)
    # Generate Gaussian noise with the same shape as the input image
    gauss = np.random.normal(0, 1, image.shape).astype(np.float32)
    # Add speckle noise to the image
    noise = image + image * gauss
    # Clip the pixel values to be in the proper range
    noise = np.clip(noise, 0, 255)
    # Convert the resultant noisy image back to the original data type of input image
    noisy_image = noise.astype(np.uint8)  # Use np.uint8 for compatibility with pytesseract
    return noisy_image


def speckle_noise_comparison(image, image_name):
    noisy_image = add_speckle_noise(image)
    noisy_text = pytesseract.image_to_string(noisy_image)
    # print("NOISY TEXT: ", noisy_text)
    ground_truth = getGroundTruthForImage(image)
    output = [
        "Image name: " + image_name + "\n",
        # "Text after adding gaussian noise: " + noisy_text + "\n",
        # "Ground truth: " + ground_truth + "\n",
        "Matched: " + str(compare_text_with_ground_truth_remove_spaces(noisy_text, ground_truth)) + "\n",
        '\n'
    ]
    with open("output_speckle_noise.txt", "a") as file:
        file.writelines(output)


speckle_noise_comparison(text_image, text_image_name)
speckle_noise_comparison(ocr_test_image, ocr_test_image_name)
speckle_noise_comparison(sample21_image, sample21_image_name)


def apply_affine_rotation_with_custom_angle(image):
    # Rotate the image by 45 degrees
    (rows, cols) = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))
    return rotated_image


def apply_affine_rotation_comparison(image, image_name):
    noisy_image = add_speckle_noise(image)
    noisy_text = pytesseract.image_to_string(noisy_image)
    ground_truth = getGroundTruthForImage(image)
    output = [
        "Image name: " + image_name + "\n",
        # "Text after adding speckle noise: " + noisy_text + "\n",
        "Ground truth: " + ground_truth + "\n",
        "Matched: " + str(compare_text_with_ground_truth_remove_spaces(noisy_text, ground_truth)) + "\n",
        '\n'
    ]
    with open("output_affine_rotation_noise.txt", "a") as file:
        file.writelines(output)


# apply_affine_rotation_comparison(text_image, text_image_name)
# apply_affine_rotation_comparison(ocr_test_image, ocr_test_image_name)
# apply_affine_rotation_comparison(sample21_image, sample21_image_name)


def apply_affine_vertical_shear(image):
    # Shear the image vertically
    (rows, cols) = image.shape[:2]
    M = np.float32([[1, 0.5, 0], [0, 1, 0]])
    sheared_image = cv2.warpAffine(image, M, (cols, rows))
    return sheared_image


def apply_affine_vertical_shear_comparison(image, image_name):
    noisy_image = apply_affine_vertical_shear(image)
    noisy_text = pytesseract.image_to_string(noisy_image)
    ground_truth = getGroundTruthForImage(image)
    output = [
        "Image name: " + image_name + "\n",
        # "Text after adding speckle noise: " + noisy_text + "\n",
        # "Ground truth: " + ground_truth + "\n",
        "Matched: " + str(compare_text_with_ground_truth_remove_spaces(noisy_text, ground_truth)) + "\n",
        '\n'
    ]
    with open("output_affine_vertical_shear_noise.txt", "a") as file:
        file.writelines(output)


apply_affine_vertical_shear_comparison(text_image, text_image_name)
apply_affine_vertical_shear_comparison(ocr_test_image, ocr_test_image_name)
apply_affine_vertical_shear_comparison(sample21_image, sample21_image_name)


def apply_affine_horizontal_shear(image):
    # Shear the image horizontally
    (rows, cols) = image.shape[:2]
    M = np.float32([[1, 0, 0], [0.5, 1, 0]])
    sheared_image = cv2.warpAffine(image, M, (cols, rows))
    return sheared_image


def apply_affine_horizontal_shear_comparison(image, image_name):
    noisy_image = apply_affine_horizontal_shear(image)
    noisy_text = pytesseract.image_to_string(noisy_image)
    ground_truth = getGroundTruthForImage(image)
    output = [
        "Image name: " + image_name + "\n",
        # "Text after adding speckle noise: " + noisy_text + "\n",
        # "Ground truth: " + ground_truth + "\n",
        "Matched: " + str(compare_text_with_ground_truth_remove_spaces(noisy_text, ground_truth)) + "\n",
        '\n'
    ]
    with open("output_affine_horizontal_shear_noise.txt", "a") as file:
        file.writelines(output)


apply_affine_horizontal_shear_comparison(text_image, text_image_name)
apply_affine_horizontal_shear_comparison(ocr_test_image, ocr_test_image_name)
apply_affine_horizontal_shear_comparison(sample21_image, sample21_image_name)


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


def gaussian_blur_comparison(image, image_name):
    noisy_image = blur_image_1(image)
    noisy_text = pytesseract.image_to_string(noisy_image)
    ground_truth = getGroundTruthForImage(image)
    output = [
        "Image name: " + image_name + "\n",
        # "Text after adding speckle noise: " + noisy_text + "\n",
        # "Ground truth: " + ground_truth + "\n",
        "Matched: " + str(compare_text_with_ground_truth_remove_spaces(noisy_text, ground_truth)) + "\n",
        '\n'
    ]
    with open("output_gaussian_blur.txt", "a") as file:
        file.writelines(output)


gaussian_blur_comparison(text_image, text_image_name)
gaussian_blur_comparison(ocr_test_image, ocr_test_image_name)
gaussian_blur_comparison(sample21_image, sample21_image_name)


def blur_image_2(image):
    # Apply median blur to the image
    blurred_image = cv2.medianBlur(image, 5)
    return blurred_image


def median_blur_comparison(image, image_name):
    noisy_image = blur_image_2(image)
    noisy_text = pytesseract.image_to_string(noisy_image)
    ground_truth = getGroundTruthForImage(image)
    output = [
        "Image name: " + image_name + "\n",
        # "Text after adding speckle noise: " + noisy_text + "\n",
        # "Ground truth: " + ground_truth + "\n",
        "Matched: " + str(compare_text_with_ground_truth_remove_spaces(noisy_text, ground_truth)) + "\n",
        '\n'
    ]
    with open("output_median_blur.txt", "a") as file:
        file.writelines(output)


median_blur_comparison(text_image, text_image_name)
median_blur_comparison(ocr_test_image, ocr_test_image_name)
median_blur_comparison(sample21_image, sample21_image_name)


def blur_image_3(image):
    # Apply bilateral filter to the image
    blurred_image = cv2.bilateralFilter(image, 9, 75, 75)
    return blurred_image


def bilateral_filter_blur_comparison(image, image_name):
    noisy_image = blur_image_3(image)
    noisy_text = pytesseract.image_to_string(noisy_image)
    ground_truth = getGroundTruthForImage(image)
    output = [
        "Image name: " + image_name + "\n",
        # "Text after adding speckle noise: " + noisy_text + "\n",
        # "Ground truth: " + ground_truth + "\n",
        "Matched: " + str(compare_text_with_ground_truth_remove_spaces(noisy_text, ground_truth)) + "\n",
        '\n'
    ]
    with open("output_bilateral_filter_blur.txt", "a") as file:
        file.writelines(output)


bilateral_filter_blur_comparison(text_image, text_image_name)
bilateral_filter_blur_comparison(ocr_test_image, ocr_test_image_name)
bilateral_filter_blur_comparison(sample21_image, sample21_image_name)

def sharpen_image(image):
    # Apply sharpening to the image
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

def sharpen_comparison(image, image_name):
    noisy_image = sharpen_image(image)
    noisy_text = pytesseract.image_to_string(noisy_image)
    ground_truth = getGroundTruthForImage(image)
    output = [
        "Image name: " + image_name + "\n",
        # "Text after adding speckle noise: " + noisy_text + "\n",
        # "Ground truth: " + ground_truth + "\n",
        "Matched: " + str(compare_text_with_ground_truth_remove_spaces(noisy_text, ground_truth)) + "\n",
        '\n'
    ]
    with open("output_sharpen.txt", "a") as file:
        file.writelines(output)

sharpen_comparison(text_image, text_image_name)
sharpen_comparison(ocr_test_image, ocr_test_image_name)
sharpen_comparison(sample21_image, sample21_image_name)

def apply_erosion(image):
    # Apply erosion to the image
    kernel = np.ones((5, 5), np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=1)
    return eroded_image

def apply_erosion_comparison(image, image_name):
    noisy_image = apply_erosion(image)
    noisy_text = pytesseract.image_to_string(noisy_image)
    ground_truth = getGroundTruthForImage(image)
    output = [
        "Image name: " + image_name + "\n",
        # "Text after adding speckle noise: " + noisy_text + "\n",
        # "Ground truth: " + ground_truth + "\n",
        "Matched: " + str(compare_text_with_ground_truth_remove_spaces(noisy_text, ground_truth)) + "\n",
        '\n'
    ]
    with open("output_erosion.txt", "a") as file:
        file.writelines(output)

apply_erosion_comparison(text_image, text_image_name)
apply_erosion_comparison(ocr_test_image, ocr_test_image_name)
apply_erosion_comparison(sample21_image, sample21_image_name)

def apply_dilation(image):
    # Apply dilation to the image
    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    return dilated_image


import matplotlib.pyplot as plt

def read_results(file_name):
    results = []
    try:
        with open(file_name, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith("Matched:"):
                    matched_percentage = float(line.split(":")[1].strip())
                    results.append(matched_percentage)
    except FileNotFoundError:
        print(f"File not found: {file_name}")
    except ValueError as e:
        print(f"Value error in file {file_name}: {e}")
    return results

def plot_results(file_name, results):
    if not results:
        print(f"No results to plot for file: {file_name}")
        return
    plt.figure()
    plt.plot(results, marker='o')
    plt.title(f'Results from {file_name}')
    plt.xlabel('Test Case')
    plt.ylabel('Matched Percentage')
    plt.ylim(0, 100)
    plt.grid(True)
    plt.savefig(f'{file_name}.png')
    plt.close()

files = [
    'output_median_blur.txt',
    'output_bilateral_filter_blur.txt',
    'output_salt_pepper.txt',
    'output_gaussian_noise.txt',
    'output_speckle_noise.txt',
    'output_affine_rotation_noise.txt',
    'output_affine_vertical_shear_noise.txt',
    'output_affine_horizontal_shear_noise.txt',
    'output_gaussian_blur.txt',
    'output_sharpen.txt',
    'output_erosion.txt'
]

for file in files:
    results = read_results(file)
    plot_results(file, results)