from ultralytics import YOLO
import cv2
import cvzone
import math
import time

cap = cv2.VideoCapture("cars.mp4") 
 
model = YOLO("yolov8l.pt")
 
classNames = ['Helmet']
#mask = cv2.imread("mask.png")
#mask = cv2.resize(mask, (1280, 720))test

while True:
 
    success, img = cap.read()
    #imgRegion = cv2.bitwise_and(img, mask)
    
    results = model(img, stream=True) # results including class, score, and location
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            currentClass = classNames[cls]

            if currentClass == "Helmet"  and conf > 0.5:
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1, offset=3)
                cvzone.cornerRect(img, (x1, y1, w, h))

    cv2.imshow("ImageRegion", img)
    cv2.waitKey(1)