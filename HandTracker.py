import sys
import cv2 as cv
import mediapipe as mp
import time
import numpy as np
import pyttsx3
import joblib
from collections import deque
import concurrent.futures

# Check command line arguments
if len(sys.argv) != 2 or sys.argv[1] not in ['--letters', '--l', '--numbers', '--n']:
    print("Usage: python HandTracker.py --letters | --numbers")
    sys.exit(1)

# Load the appropriate KNN model and label encoder based on the argument
if sys.argv[1] in ['--letters', '--l']:
    knn = joblib.load('knn_letteronly_model.pkl')
    label_encoder = joblib.load('label_letteronly_encoder.pkl')
elif sys.argv[1] in ['--numbers', '--n']:
    knn = joblib.load('knn_numbersonly_model.pkl')
    label_encoder = joblib.load('label_numbersonly_encoder.pkl')

# Load video capture
cap = cv.VideoCapture(0)

# Loading MediaPipe's Hands detection model
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)

# Initializing Queue to keep track of predicted characters to read out
speech_queue = deque()

# Variables to keep track of the predicted character and time
last_predicted_character = None
last_dictated_character = None
last_predicted_time = time.time()

# Function to handle text-to-speech in the background
def speak_character(character):
    engine.say(character)
    engine.runAndWait()

# Create a ThreadPoolExecutor for running text-to-speech tasks in the background
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# Stored phrase
phrase = ""
current_future = None
current_letter = None

def print_phrase():
    print("Phrase: " + '"' + phrase + '"')

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    current_time = time.time()
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

            # Check if the predicted character has been the same for at least 1 second
            if predicted_character == last_predicted_character:
                if current_time - last_predicted_time >= 1.5:
                    if current_future is None or current_future.running() is None or current_future.done(): #Check if the phrase is done
                        if current_letter is None or current_letter.running() is None or current_letter.done(): #Check if the letter is done
                            #print(predicted_character)
                            speech_queue.append(predicted_character)
                            phrase += predicted_character
                            print_phrase()
                            last_predicted_time = current_time  # Reset the timer after speaking
                            last_dictated_character = predicted_character
            else:
                last_predicted_character = predicted_character
                last_predicted_time = current_time
    elif current_time - last_predicted_time >= 1.5 and phrase != "" and last_predicted_character != " ":
        print_phrase()
        last_predicted_character = " "
        phrase += last_predicted_character

    # Process the speech queue synchronously
    if speech_queue:
        character = speech_queue.popleft()
        if character is not None:
            current_letter = executor.submit(speak_character, character)

    cv.imshow("Image", img)
    key = cv.waitKey(1) & 0xFF

    # Reads out the stored phrase
    if key == ord('r') and phrase:
        print("Reading out the stored phrase...")
        print_phrase()
        current_future = executor.submit(speak_character, phrase)
        phrase = ""

    # Erases the last character in the stored phrase
    if key == 8 and len(phrase) >= 1:
        phrase = phrase[:-1]
        print_phrase()

    # Clears the stored phrase
    if key == ord('c') and phrase != "":
        print("Clearing phrase...")
        phrase = ""

    if key == ord('q'):
        print("Exiting program...")
        time.sleep(0.5)
        speech_queue.append(None)

        # Shutdown the executor
        executor.shutdown(wait=True, cancel_futures=True)
        break

engine.stop()
cap.release()
cv.destroyAllWindows()
sys.exit(0)