import cv2
import sys
import keras
import numpy as np
cap = cv2.VideoCapture(0)
ret,frame = cap.read()

model = keras.models.load_model('./cnn_model')

angry = cv2.imread('./emoji/angry.png')
disgust = cv2.imread('./emoji/disgust.png')
fear = cv2.imread('./emoji/fear.png')
happy = cv2.imread('./emoji/happy.png')
sad = cv2.imread('./emoji/sad.png')
surprise = cv2.imread('./emoji/surprise.png')
neutral = cv2.imread('./emoji/neutral.png')

expr_dict = {0:angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral}
while ret:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        face = gray[y:y+h,x:x+w]
        std_face = np.expand_dims(np.expand_dims(cv2.resize(face, (48,48)), axis=-1), axis=0)

        prediction = model.predict(std_face)[0]
        cv2.imshow('emoji',expr_dict[np.argmax(prediction)])

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
