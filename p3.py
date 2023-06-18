import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime 
import csv

now=datetime.now()
current_date=now.strftime("%Y-%m-%d")

f=open(current_date+'.csv','w+',newline='')
lnwriter = csv.writer(f)

#to read the files from purticular folder and appending those to a list called myList
path=r"C:\Users\91961\Desktop\face_detection\photos"
images=[]
classNames=[]
myList=os.listdir(path)
print(myList)

#to get the names of images 
for cl in myList:
	curImg=cv2.imread(f'{path}/{cl}') 
	images.append(curImg) 
	classNames.append(os.path.splitext(cl)[0]) #to read the names of the images
print(classNames)


def findEncodings(images):
	encodeList =[]
	for img in images :
		img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		encode=face_recognition.face_encodings(img)[0]
		encodeList.append(encode)
	return encodeList

def markAttendence(name):
	with open(current_date+'.csv','r+') as f :
		myDataList = f.readlines()
		nameList =[]
		for line in myDataList:
			entry=line.split(',')
			nameList.append(entry[0])
		if name not in nameList:
			now = datetime.now()
			dtString= now.strftime('%H:%M:%S')
			f.writelines(f'\n{name},{dtString}')

encodeListKnown=findEncodings(images)
print(len(encodeListKnown))
print("encoding completed")

#to take video captures
cap=cv2.VideoCapture(0)

while True:
	success,img=cap.read()
	imgS=cv2.resize(img,(0,0),None,0.25,0.25)
	imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

	facesCurFrame = face_recognition.face_locations(imgS)
	encodesCurFrame=face_recognition.face_encodings(imgS,facesCurFrame)

	for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
		matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
		faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
		print(faceDis)
		matchIndex=np.argmin(faceDis)


		if matches[matchIndex]:
			name=classNames[matchIndex].upper()
			y1,x2,y2,x1=faceLoc
			y1,x2,y2,x1= y1*4,x2*4,y2*4,x1*4
			cv2.rectangle(img,(x1,y1),(x2,y2),(0.255,0),2)
			cv2.rectangle(img,(x1,y2-35),(x2,y2),(0.255,0),cv2.FILLED)
			cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
			markAttendence(name)

	cv2.imshow('Webcam',img)
	cv2.waitKey(1)