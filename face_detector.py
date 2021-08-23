import cv2

#read an Image
img = cv2.imread('Car.jpg', 1)
new_img= cv2.resize(img, (int(img.shape[0]/4), int(img.shape[1] /4)))

#search for faces in Image
face_cascade = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")

#make grey scale image
gray_image = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
#cv2.imshow('Gray',gray_image)

#search faces coordinates in image
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.05, minNeighbors=10)
#print(faces)

for x,y,w,h in faces:
   img = cv2.rectangle(new_img, (x, y), (x+w, y+h), (0, 255, 0), 3)

cv2.imshow("Detected_faces", new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



