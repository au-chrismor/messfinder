#! /usr/bin/python3
#
import cv2
import os
import numpy as np
import pickle


# ### Training Data

print('Loading Subjects...')
subjects = ['']
subjects.append('Person 1')
subjects.append('Person 2')
# and so on...

print('Done')

#
# Save off the subject list
#
with open('subjects.dat', 'wb') as fp:
	pickle.dump(subjects, fp)



# In[3]:

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


def prepare_training_data(data_folder_path):
    
    #------STEP-1--------
    #get the directories (one directory for each subject) in data folder
    dirs = sorted(os.listdir(data_folder_path))
    
    #list to hold all subject faces
    faces = []
    #list to hold labels for all subjects
    labels = []
    
    #let's go through each directory and read images within it
    for dir_name in dirs:
        
        print(dir_name)

        #our subject directories start with letter 's' so
        #ignore any non-relevant directories if any
        if not dir_name.startswith('s'):
            continue;
            
        #------STEP-2--------
        #extract label number of subject from dir_name
        #format of dir name = slabel
        #, so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace('s', ''))
        
        #build path of directory containin images for current subject subject
        #sample subject_dir_path = 'training-data/s1'
        subject_dir_path = data_folder_path + '/' + dir_name
        
        #get the images names that are inside the given subject directory
        subject_images_names = sorted(os.listdir(subject_dir_path))
        
        #------STEP-3--------
        #go through each image name, read image, 
        #detect face and add face to list of faces
        for image_name in subject_images_names:
            
            #ignore system files like .DS_Store
            if image_name.startswith('.'):
                continue;
            
            #build image path
            #sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + '/' + image_name
#            print(image_path)

            #read image
            image = cv2.imread(image_path)
            
            #display an image window to show the image 
            #cv2.imshow('Training on image...', cv2.resize(image, (400, 500)))
            #cv2.waitKey(100)
            
            #detect face
            face, rect = detect_face(image)
            
            #------STEP-4--------
            #for the purpose of this tutorial
            #we will ignore faces that are not detected
            if face is not None:
                #add face to list of faces
                faces.append(face)
                #add label for this face
                labels.append(label)
            
#    cv2.destroyAllWindows()
#    cv2.waitKey(1)
#    cv2.destroyAllWindows()
    
    return faces, labels


#let's first prepare our training data
#data will be in two lists of same size
#one list will contain all the faces
#and other list will contain respective labels for each face
print('Preparing data...')
faces, labels = prepare_training_data('training-data')
print('Data prepared')

#print total faces and labels
print('Total facets: ', len(faces))
print('Total labels: ', len(labels))

print('Saving')
with open('facets.dat', 'wb') as fp:
	pickle.dump(faces, fp)
with open('labels.dat', 'wb') as fp:
	pickle.dump(labels, fp)
#
# with open('facets.dat', 'rb') as fp:
#	faces = pickle.load(fp)


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

