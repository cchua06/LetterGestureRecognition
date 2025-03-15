import sys

# Check command line arguments
if len(sys.argv) != 2 or sys.argv[1] not in ['--letters', '--numbers']:
    print("Usage: python HandTracker.py --letters | --numbers")
    sys.exit(1)

import cv2 as cv
import mediapipe as mp
import time
import numpy as np
import joblib

# Load the appropriate KNN model and label encoder based on the argument
if sys.argv[1] == '--letters':
    knn = joblib.load('knn_letteronly_model.pkl')
    label_encoder = joblib.load('label_letteronly_encoder.pkl')
elif sys.argv[1] == '--numbers':
    knn = joblib.load('knn_numbersonly_model.pkl')
    label_encoder = joblib.load('label_numbersonly_encoder.pkl')

# Load video capture
cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLandmark in results.multi_hand_landmarks:
            landmark_locs = []
            mpDraw.draw_landmarks(img, handLandmark, mpHands.HAND_CONNECTIONS)

            h, w, c = img.shape
            landmarks = np.array([(lm.x * w, lm.y * h) for lm in handLandmark.landmark])
            x_min, y_min = np.min(landmarks, axis=0)
            x_max, y_max = np.max(landmarks, axis=0)

            # Normalize the landmarks to the range [0, 1]
            width = x_max - x_min
            height = y_max - y_min
            normalized_landmarks = (landmarks - [x_min, y_min]) / [width, height]
            
            landmark_locs = normalized_landmarks.tolist()
            
            # Flatten the landmark locations for prediction
            landmark_locs_flat = np.array(landmark_locs).flatten().reshape(1, -1)
            
            # Predict the letter or number
            predicted_label = knn.predict(landmark_locs_flat)
            predicted_character = label_encoder.inverse_transform(predicted_label)[0].upper()
            
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

            cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv.putText(img, f"Character: {predicted_character}", (x_min, y_min - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (139, 0, 0), 2)

    cv.imshow("Image", img)
    key = cv.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Exiting")
        break

cap.release()
cv.destroyAllWindows()