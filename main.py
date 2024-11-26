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
    cv2.imshow("original image", image)
    amount_for_salt_pepper = 0.01
    noisy_image = salt_pepper_noise(image, amount_for_salt_pepper)
    cv2.imshow("noisy image", noisy_image)
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
    cv2.imshow("original image", image)
    noisy_image = add_gaussian_noise(image)
    cv2.imshow("noisy image", noisy_image)
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
    cv2.imshow("original image", image)
    noisy_image = add_speckle_noise(image)
    cv2.imshow("noisy image", noisy_image)
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
    cv2.imshow("original image", image)
    noisy_image = add_speckle_noise(image)
    cv2.imshow("noisy image", noisy_image)
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
    cv2.imshow("original image", image)
    noisy_image = apply_affine_vertical_shear(image)
    cv2.imshow("noisy image", noisy_image)
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
    cv2.imshow("original image", image)
    noisy_image = apply_affine_horizontal_shear(image)
    cv2.imshow("noisy image", noisy_image)
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


#
# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
#                 help="path to input image to be OCR'd")
# ap.add_argument("-p", "--preprocess", type=str, default="thresh",
#                 help="type of preprocessing to be done")
# args = vars(ap.parse_args())

# load the example image and convert it to grayscale
# check to see if we should apply thresholding to preprocess the
# # image
# if args["preprocess"] == "thresh":
#     gray = cv2.threshold(gray, 0, 255,
#                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# # make a check to see if median blurring should be done to remove
# # noise
# elif args["preprocess"] == "blur":
#     gray = cv2.medianBlur(gray, 3)


# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
# filename = "{}.png".format(os.getpid())
# cv2.imwrite(filename, gray)

# load the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary file
# text = pytesseract.image_to_string(Image.open(filename))
# os.remove(filename)
# # print(text)
# ground_truth = "The quick brown fox jumps over the lazy dog"


def apply_each_noise_method_and_compare_the_text_with_ground_truth(image):
    # Apply each noise method and compare the text with the ground truth
    # 1. Salt and Pepper Noise
    salt_pepper_comparison_text(image)
    # 2. Gaussian Noise
    # noisy_image = add_gaussian_noise(gray)
    # filename = "{}.png".format(os.getpid())
    # cv2.imwrite(filename, noisy_image)
    # noisy_text = pytesseract.image_to_string(Image.open(filename))
    # os.remove(filename)
    # with open("", "w") as file:
    #     file.write("Gaussian Noise: " + noisy_text + "\n")
    #     file.write("Matched: " + str(compare_text_with_ground_truth_remove_spaces(noisy_text, ground_truth)) + "\n")
    # # 3. Speckle Noise
    # noisy_image = add_speckle_noise(gray)
    # filename = "{}.png".format(os.getpid())
    # cv2.imwrite(filename, noisy_image)
    # noisy_text = pytesseract.image_to_string(Image.open(filename))
    # os.remove(filename)
    # with open("", "w") as file:
    #     file.write("Speckle Noise: " + noisy_text + "\n")
    #     file.write("Matched: " + str(compare_text_with_ground_truth_remove_spaces(noisy_text, ground_truth)) + "\n")
    # # 4. Affine Rotation
    # rotated_image = apply_affine_rotation_with_custom_angle(gray)
    # filename = "{}.png".format(os.getpid())
    # cv2.imwrite(filename, rotated_image)
    # rotated_text = pytesseract.image_to_string(Image.open(filename))
    # os.remove(filename)
    # with open("output_rotation.txt", "w") as file:
    #     file.write("Affine Rotation: " + rotated_text + "\n")
    #     file.write(
    #         "Matched: " + str(compare_text_with_ground_truth_remove_spaces(rotated_text, ground_truth)) + "\n")
    # # 5. Affine Vertical Shear
    # sheared_image = apply_affine_vertical_shear(gray)
    # filename = "{}.png".format(os.getpid())
    # cv2.imwrite(filename, sheared_image)
    # sheared_text = pytesseract.image_to_string(Image.open(filename))
    # os.remove(filename)
    # with open("output_vertical_shear.txt", "w") as file:
    #     file.write("Affine Vertical Shear: " + sheared_text + "\n")
    #     file.write(
    #         "Matched: " + str(compare_text_with_ground_truth_remove_spaces(sheared_text, ground_truth)) + "\n")
    # 6. Affine Horizontal Shear
    # sheared_image = apply_affine_horizontal_shear(gray)
    # filename = "{}.png".format(os.getpid())
    # cv2.imwrite(filename, sheared_image)
    # sheared_text = pytesseract.image_to_string(Image.open(filename))
    # os.remove(filename)
    # with open("output_horizontal_shear.txt", "w") as file:
    #     file.write("Affine Horizontal Shear: " + sheared_text + "\n")
    #     file.write(
    #         "Matched: " + str(compare_text_with_ground_truth_remove_spaces(sheared_text, ground_truth)) + "\n")
    # 7. Resize by Keeping Aspect Ratio
    # resized_image = resize_by_keeping_aspect_ratio(gray, width=500)
    # filename = "{}.png".format(os.getpid())
    # cv2.imwrite(filename, resized_image)
    # resized_text = pytesseract.image_to_string(Image.open(filename))
    # os.remove(filename)
    # with open("output_resize_aspect_ratio.txt", "w") as file:
    #     file.write("Resize by Keeping Aspect Ratio: " + resized_text + "\n")
    #     file.write(
    #         "Matched: " + str(compare_text_with_ground_truth_remove_spaces(resized_text, ground_truth)) + "\n")
    # 5. Resize by Distorting Aspect Ratio
    # resized_image = resize_by_distorting_aspect_ratio(gray, width=500, height=500)
    # filename = "{}.png".format(os.getpid())
    # cv2.imwrite(filename, resized_image)
    # resized_text = pytesseract.image_to_string(Image.open(filename))

#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
