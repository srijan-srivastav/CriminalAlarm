import numpy as mp
import cv2,time
import pickle

labels={}
with open("labels.pickle",'rb') as f:
	temp_labels = pickle.load(f)
	labels={v:k for k,v in temp_labels.items()}

face_cascade=cv2.CascadeClassifier('/home/srijan/Desktop/CriminalAlarm/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

cap=cv2.VideoCapture(0)

while(True):


	ret,frame =cap.read()
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)


	for(x,y,w,h) in faces:
		#print(x,y,w,h)
		gray_area = gray[y:y+h, x:x+w]
		idd,conf =recognizer.predict(gray_area)
		if conf>=45 and conf<=85:
			print(labels[idd])
			font=cv2.FONT_HERSHEY_DUPLEX
			name=labels[idd]
			color=(255,255,255)
			stroke=2
			if name=="srijan":
				color=(0,0,255)
				name+=" (wanted)"
				stroke=1
				name=name.upper()
				cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
				print('\a')


			cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)

		color=(0,255,0)
		stroke=2
		width=x+w
		height=y+h
		cv2.rectangle(frame,(x,y),(width,height),color,stroke)


	cv2.imshow('frame',frame)
	key=cv2.waitKey(1)
	if key == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()