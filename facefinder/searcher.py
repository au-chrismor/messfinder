#! /usr/bin/python3
#
import cv2
import os
import numpy as np
import pickle


# ### Training Data

print('Loading Subjects...')

with open('subjects.dat', 'rb') as fp:
    subjects = pickle.load(fp)

print('Done')

#function to detect face using OpenCV
def detect_face(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
    
    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]
    
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]

print('Preparing data...')
with open('facets.dat', 'rb') as fp:
    faces = pickle.load(fp)

with open('labels.dat', 'rb') as fp:
    labels = pickle.load(fp)

print('Data prepared')

#print total faces and labels
print('Total facets: ', len(faces))
print('Total labels: ', len(labels))

print('Training Recogniser...')
#create our LBPH face recognizer 
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#or use EigenFaceRecognizer by replacing above line with 
#face_recognizer = cv2.face.EigenFaceRecognizer_create()

#or use FisherFaceRecognizer by replacing above line with 
#face_recognizer = cv2.face.FisherFaceRecognizer_create()


#train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))


#function to draw rectangle on image 
#according to given (x, y) coordinates and 
#given width and heigh
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
#function to draw text on give image starting from
#passed (x, y) coordinates. 
def draw_text(img, text, x, y):
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(img, text, (x, y), font, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
    return



#this function recognizes the person in image passed
#and draws a rectangle around detected face with name of the 
#subject
def predict(test_img):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img)

    #predict the image using our face recognizer 
    if face is None:
        return
    label, confidence = face_recognizer.predict(face)
    print('Label: ', label)
    print('Confidence: ', confidence)
    #get name of respective label returned by face recognizer
    try:
        label_text = subjects[label]
    except:
        label_text = "Unknown - bad index"
    
    #draw a rectangle around face detected
    draw_rectangle(img, rect)
    #draw name of predicted person
    draw_text(img, label_text, rect[0], rect[1]-5)

    return img

print('Predicting images...')

#load test images
tests = sorted(os.listdir('test-data'))
for test_img_name in tests:
    print(test_img_name)
    test_img = cv2.imread('test-data/' + test_img_name)

#perform a prediction
    if test_img is not None:
        predicted_img1 = predict(test_img)
    print('Prediction complete')

#display both images
#cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))
#cv2.imshow(subjects[2], cv2.resize(predicted_img2, (400, 500)))
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#cv2.waitKey(1)
#cv2.destroyAllWindows()
