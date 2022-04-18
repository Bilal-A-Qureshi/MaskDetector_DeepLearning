import cv2
import numpy as np
from keras.models import load_model

text_dict={0: "Mask on", 1: "No mask"}
rect_color_dict={0:(0,255,0),1:(0,0,255)}
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

model = load_model('face_mask_detection_model.h5')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 3)

    for (x, y, w, h) in faces:

        face_img = gray[y:y+w, x:x+w]
        resized_img = cv2.resize(face_img,(112,112))
        normalized_img=resized_img/255.0
        reshaped_img=np.reshape(normalized_img,(1,112,112,1))
        result=model.predict(reshaped_img)

        label=np.argmax(result,axis=1)[0]

        print(label)

        cv2.rectangle(img,(x,y),(x+w,y+h),rect_color_dict[label],2)
        cv2.rectangle(img, (x, y-40), (x + w, y), rect_color_dict[label], -1)
        cv2.putText(img, text_dict[label], (x,y-10),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,0),2)


        #cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        if label ==0:
            print("Mask Is On")
        else:
            print("No mask is On")


    cv2.imshow('img', img)
    cv2.waitKey(1)