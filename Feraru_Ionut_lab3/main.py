import cv2
import numpy as np
import matplotlib.pyplot as plt

# figure = plt.figure(figsize=(13,10))
# rows = 3
# columns = 4
#
# def plot_image(image, title, place_of_the_image):
#     figure.add_subplot(rows, columns, place_of_the_image)
#     plt.imshow(image)
#     plt.axis('on')
#     plt.title(title)

def detect_skin_pixels_1(image, position):
    skin_image = np.zeros_like(image)
    (row, col) = image.shape[0:2]
    for i in range(row):
        for j in range(col):
            b = image[i, j][0]
            g = image[i, j][1]
            r = image[i, j][2]
            if r > 95 and g > 40 and b > 20 and max(r, g, b) - min(r, g, b) > 15 and abs(r - g) > 15 and r > g and r > b:
                skin_image[i, j] = (255, 255, 255)
            else:
                skin_image[i, j] = (0, 0, 0)

    # plot_image(skin_image, "skin_image"+str(position), position)
    return skin_image

# for i in range(1, 12):
#     image = cv2.imread(f'skin_images/skin/{i}.jpg')
#     detect_skin_pixels_1(image, i)

def detect_skin_pixels_2(image, position):
    skin_image = np.zeros_like(image)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    (row, col) = image.shape[0:2]
    for i in range(row):
        for j in range(col):
            h = hsv_image[i, j][0]
            s = hsv_image[i, j][1]
            v = hsv_image[i, j][2]
            #  In OpenCV, the Saturation and Value channels range from 0 to 255, not from 0 to 1
            # 0 <= h <= 50 and 0.23 <= s <= 0.68 and 0.35
            if 0 <= h <= 50 and 58 <= s <= 173 and 89 <= v <= 255:
                skin_image[i, j] = (255, 255, 255)
            else:
                skin_image[i, j] = (0, 0, 0)

    # plot_image(skin_image, "skin_image"+str(position), position)
    return skin_image

# for i in range(1, 12):
#     image = cv2.imread(f'skin_images/skin/{i}.jpg')
#     detect_skin_pixels_2(image, i)

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

    # plot_image(skin_image, "skin_image"+str(position), position)
    return skin_image


# for i in range(1, 12):
#     image = cv2.imread(f'skin_images/skin/{i}.jpg')
#     detect_skin_pixels_3(image, i)
# plt.show()

def evaluate_skin_detection(image, ground_truth_image):
    skin_image = detect_skin_pixels_3(image, 1)

    ground_truth_image_file_name = ground_truth_image

    ground_truth_image = cv2.imread(ground_truth_image)
    ground_truth_image = cv2.cvtColor(ground_truth_image, cv2.COLOR_BGR2GRAY)
    _, ground_truth_image = cv2.threshold(ground_truth_image, 127, 255, cv2.THRESH_BINARY)
    confusion_matrix = np.zeros((2, 2))
    (row, col) = skin_image.shape[0:2]
    for i in range(row):
        for j in range(col):
            if ground_truth_image[i, j] == 255 and skin_image[i, j][0] == 255:
                confusion_matrix[0, 0] += 1
            elif ground_truth_image[i, j] == 0 and skin_image[i, j][0] == 0:
                confusion_matrix[1, 1] += 1
            elif ground_truth_image[i, j] == 255 and skin_image[i, j][0] == 0:
                confusion_matrix[0, 1] += 1 # false positive
            else:
                confusion_matrix[1, 0] += 1 # false negative

    accuracy = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / (confusion_matrix[0, 0] + confusion_matrix[1, 1] + confusion_matrix[0, 1] + confusion_matrix[1, 0])
    print("Confusion Matrix: ", confusion_matrix)
    print("Accuracy: ", accuracy)
    with open('accuracy_results_3_family.txt', 'a') as f:
        f.write(f'{ground_truth_image_file_name}: {accuracy}\n')

import os

image_files = os.listdir('Pratheepan_Dataset/FamilyPhoto')
ground_truth_files = os.listdir('Ground_Truth/GroundT_FamilyPhoto')

# for image_file, ground_truth_file in zip(sorted(image_files), sorted(ground_truth_files)):
#     image = cv2.imread(f'Pratheepan_Dataset/FamilyPhoto/{image_file}')
#     print(f'Pratheepan_Dataset/FacePhoto/{image_file}')
#     ground_truth_image = f'Ground_Truth/GroundT_FamilyPhoto/{ground_truth_file}'
#     print(f'Ground_Truth/GroundT_FacePhoto/{ground_truth_file}')
#     evaluate_skin_detection(image, ground_truth_image)

with open('accuracy_results_1_family.txt', 'r') as f:
    accuracies_1 = [float(line.split(': ')[1]) for line in f.readlines()]
with open('accuracy_results_2_family.txt', 'r') as f:
    accuracies_2 = [float(line.split(': ')[1]) for line in f.readlines()]
with open('accuracy_results_3_family.txt', 'r') as f:
    accuracies_3 = [float(line.split(': ')[1]) for line in f.readlines()]

# compute the medium accuracy for each method
print("Method 1: ", np.median(accuracies_1))
print("Method 2: ", np.median(accuracies_2))
print("Method 3: ", np.median(accuracies_3))

# Plot the accuracies in a line chart
plt.figure(figsize=(10, 5))
plt.plot(range(len(accuracies_1)), accuracies_1, label='Method 1', marker='o')
plt.plot(range(len(accuracies_2)), accuracies_2, label='Method 2', marker='s', alpha=0.7)
plt.plot(range(len(accuracies_3)), accuracies_3, label='Method 3', marker='^', alpha=0.5)
plt.xlabel('Image Index')
plt.ylabel('Accuracy')
plt.title('Accuracy of Skin Detection Methods for family Photos')
plt.legend()
plt.savefig('accuracy_plot_for_family_photos.png')
plt.show()