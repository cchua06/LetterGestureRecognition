# Letter Gesture Recognition

Uses Google's prebuilt Mediapipe computer vision model to detect ASL from hand gestures. It contains two KNN models that are trained to detect either letters or numbers. The KNN models were trained using an open-source Kaggle dataset.

## Table of Contents

- [Pre-requisites](#pre-requisites)
- [How to run](#how-to-run)
- [Known bugs](#known-bugs)
- [References](#references)
- [Contact](#contact)

## Pre-requisites

Use pip to install: mediapipe, opencv-python, pyttsx3 and numpy.

## How to run

Usage: py HandTracker.py --[flag]. Use --letters to classify letters and use --numbers to classify numbers. The program stores letters classified into a phrase. Hold the letter for 1.5 seconds for the letter to be dictated and added to the phrase. Once a letter is added, you may add a space character by removing your hands from the camera's view for 1.5 seconds.

Press "r" to replay the phrase. Press "c" to clear the stored phrase. Press BACKSPACE to remove the most recent character from the phrase. 

Press "q" to exit the program.

## Known bugs

Exiting the program while the speech engine is running will cause the program to crash.

## References

[Mediapipe Gesture Recognizer](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer/python)

[ASL Kaggle Dataset](https://www.kaggle.com/datasets/ayuraj/asl-dataset/data)

[Murtaza's Workshop Hand Tracking](https://www.youtube.com/watch?v=NZde8Xt78Iw&ab_channel=Murtaza%27sWorkshop-RoboticsandAI)

## Contact

Cedric Chua - cchua06@sas.upenn.edu
Project Link - https://github.com/cchua06/LetterGestureRecognition