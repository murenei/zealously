import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_cascades = cv2.CascadeClassifier('/Users/richardmurray/Projects/AnacondaProjects/TestSpyder/opencv/cascades/haarcascade_frontalface_default.xml')


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detects faces and returns tuple of x,y coordinates and h,w for each face
    faces=face_cascades.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    
    # Print coordinates
    for (x,y,w,h) in faces:
        print(x,y,w,h)
        
        # get second set of coordinates
        x2=x+w
        y2=y+h
        
        # Get region of interest
        roi_color=frame[y:y2,x:x2]    # x1, y1, x2, y2 coordinates
        
        ### Recognise Face
        
        
        # write to an image each loop
        img='my-img.png'
        cv2.imwrite(img,roi_color)  #imgwrite
        
        ### Draw Rectangles
        
        # Set parameters
        color=(255,0,0)     # BGR 
        stroke=2
        cv2.rectangle(frame,(x,y),(x2,y2), color, stroke)
        

    # Display the resulting frame
    cv2.imshow('frame',frame)
#    cv2.imshow('grey',grey)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
