# Face and Emotion Recognition
This software recognizes person's faces and their corresponding emotions from a video or webcam feed. Powered by OpenCV, Dlib, face_recognition and Deep Learning.

## Demo
![Image](https://user-images.githubusercontent.com/22372476/47372515-920f0180-d707-11e8-9ba5-d3f51020958a.gif)


## Dependencies
- Opencv
- Dlib
- [face_recognition](https://github.com/ageitgey/face_recognition)
- Keras

## Usage
- Download a `shape_predictor_68_face_landmarks.dat` file from [here](https://drive.google.com/open?id=1hyDn8eJ5yaTVkMgdKGmoFIn48zwdvIkg) and put in the folder.
- `test` folder contain images or video that we will feed to the model.
- `images` folder contain only images of person face to perform face recognition.
- `models` contain the pre-trained model for emotion classifier.
- `emotion.py` can to run to classify emotions of person's face.
- `face-rec-emotion.py` can recognise faces and classify emotion at a time.
- face_recognition library uses the FaceNet Implementation for face recognition.For more details please visit [here](https://github.com/ageitgey/face_recognition)

`python emotion.py`

`python face-rec-emotion.py`


## To train new models for emotion classification

- Download the fer2013.tar.gz file from [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- Move the downloaded file to the datasets directory inside this repository.
- Untar the file:
`tar -xzf fer2013.tar`
- Download train_emotion_classifier.py from orriaga's repo [here](https://github.com/oarriaga/face_classification/blob/master/src/train_emotion_classifier.py)
- Run the train_emotion_classification.py file:
`python train_emotion_classifier.py`

## Deep Learning Model

The model used is from this [research paper](https://github.com/oarriaga/face_classification/blob/master/report.pdf) written by Octavio Arriaga, Paul G. Plöger, and Matias Valdenegro.

![Model](https://i.imgur.com/vr9yDaF.png?1)

## Credit

* Computer vision powered by OpenCV.
* Neural network scaffolding powered by Keras with Tensorflow.
* FaceNet [Research Paper](https://arxiv.org/pdf/1503.03832.pdf)
* Convolutional Neural Network (CNN) deep learning architecture is from this [research paper](https://github.com/oarriaga/face_classification/blob/master/report.pdf).
* Pretrained Keras model and much of the OpenCV code provided by GitHub user [oarriaga](https://github.com/oarriaga).
