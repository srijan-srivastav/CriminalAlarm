import os
import numpy as np
from PIL import Image
import cv2
import pickle
base_dir=os.path.dirname(os.path.abspath(__file__))
img_dir=os.path.join(base_dir,"images")

face_cascade=cv2.CascadeClassifier('/home/srijan/Desktop/CriminalAlarm/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

cur_id=0
label_ids={}
y_labels=[ ] 
x_train=[]

for root,dirs,files in os.walk(img_dir):
	for file in files:
		if file.endswith("jpg"):
			path=os.path.join(root,file)
			label=os.path.basename(root).lower()
			#print(label,path)
			if not label in label_ids:
				label_ids[label]=cur_id
				cur_id+=1
			idd=label_ids[label]
			#print(label_ids)
			pil_image=Image.open(path).convert("L")
			img_array=np.array(pil_image,"uint8")
			faces=face_cascade.detectMultiScale(img_array,scaleFactor=1.5,minNeighbors=5)
			for (x,y,w,h) in faces:
				roi=img_array[y:y+h,x:x+w]
				x_train.append(roi)
				y_labels.append(idd)

with open("labels.pickle",'wb') as f:
	pickle.dump(label_ids,f)
recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainer.yml")