# Letter Gesture Recognition

Uses Google's prebuilt Mediapipe computer vision model to detect ASL from hand gestures. It contains two KNN models that are trained to detect either letters or numbers. The KNN models were trained using an open-source Kaggle dataset.

## Table of Contents

- [Pre-requisites](#pre-requisites)
- [How to run](#how-to-run)
- [References](#references)
- [Contact](#contact)

## Pre-requisites

Use pip to install: mediapipe, opencv-python and numpy.

## How to run

Usage: py HandTracker.py --[flag]. Use --letters to classify letters and use --numbers to classify numbers.

## References

[Mediapipe Gesture Recognizer](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer/python)

[ASL Kaggle Dataset](https://www.kaggle.com/datasets/ayuraj/asl-dataset/data)

[Murtaza's Workshop Hand Tracking](https://www.youtube.com/watch?v=NZde8Xt78Iw&ab_channel=Murtaza%27sWorkshop-RoboticsandAI)

## Contact

Cedric Chua - cchua06@sas.upenn.edu
Project Link - https://github.com/cchua06/LetterGestureRecognition