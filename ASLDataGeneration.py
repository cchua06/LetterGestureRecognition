import numpy as np
import pandas as pd
import pathlib
import mediapipe as mp
import cv2 as cv

dataset_path = pathlib.Path('asl_dataset')

dictionary = {}

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode = True, max_num_hands = 1)

# Iterate over each folder in the dataset path
for label_folder in dataset_path.iterdir():
    if label_folder.is_dir():
        label = label_folder.name

        #Skip numbers
        if label.isdigit():
            continue

        landmarks_list = []
        
        # Iterate over each image file in the label folder
        for image_file in label_folder.iterdir():
            if image_file.suffix in ['.jpeg']:
                img = cv.imread(str(image_file))
                imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                results = hands.process(imgRGB)

                if results.multi_hand_landmarks:
                    for handLandmark in results.multi_hand_landmarks:
                        h, w, c = img.shape
                        landmarks = np.array([(lm.x * w, lm.y * h) for lm in handLandmark.landmark])
                        x_min, y_min = np.min(landmarks, axis=0)
                        x_max, y_max = np.max(landmarks, axis=0)

                        # Normalize the landmarks to the range [0, 1]
                        width = x_max - x_min
                        height = y_max - y_min
                        normalized_landmarks = (landmarks - [x_min, y_min]) / [width, height]
                        
                        landmarks_list.append(normalized_landmarks)
        
        dictionary[label] = landmarks_list

# Prepare the data for training
X = []
y = []

for label, landmarks in dictionary.items():
    for landmark in landmarks:
        X.append(landmark)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Save the data to .npz files
#np.savez('cleaned_data/asl_numbersonly_data.npz', X = X, y = y)
np.savez('cleaned_data/asl_letteronly_data.npz', X = X, y = y)