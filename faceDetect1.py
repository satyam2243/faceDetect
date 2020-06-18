import cv2
import numpy as np

#global face_classifier
face_classifier = cv2.CascadeClassifier('C:/Users/sattu/AppData/Local/Programs/Python/Python37-32/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')


# extract face feature
def face_extractor(img):  # function layout (function call in main program)
   # face_classifier = cv2.CascadeClassifier('C:/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Work on grace because its easy
    #print(gray)
    faces = face_classifier.detectMultiScale(gray,1.3,5)  # pass image sample

    if faces is():
        return None
    for (x, y, w, h) in faces:  # w=width changes & h= height changes
        cropped_face = img[y:y + h, x:x + w]
    return cropped_face


cap = cv2.VideoCapture(0)  # video capture function
count = 0  # help in counting



while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count+= 1                          # to passing the extractor through the camera then we are use
        face = cv2.resize(face_extractor(frame),(300, 300))  # resize the face because the size of camera is equal to face
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # to convert the in grayscale

        file_name_path = 'F:/faces/user' + str(count) + '.jpg'

        cv2.imwrite(file_name_path, face)

        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0, 255, 0),2)  # origin point where test make a start
        cv2.imshow('Face Cropper',face)  # output
    else:
        print("Face not Found")
        pass

    if cv2.waitKey(1) == 13 or count==100:
        break

cap.release()  # close the camera
cv2.destroyAllWindows()
print('collecting sample complete!!!')

