import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)

while True:
    sucess, img=cap.read()
    img,faces = detector.findFaceMesh(img,draw=False)
    if faces:
        face=faces[0]
        pointLeft = face[145]
        pointRight = face[374]
        # cv2.circle(img,pointLeft,5,(255,0,0),cv2.FILLED)
        # cv2.circle(img, pointRight, 5, (255, 0, 0), cv2.FILLED)
        # cv2.line(img,pointLeft,pointRight,(0,0,0),3)
        w,_ = detector.findDistance(pointLeft,pointRight)
        W = 6.3
        f = 668
        d = (W*f)/w
        cvzone.putTextRect(img,f'Distance from camera: ~{int(d)}cm',(face[10][1]-75,face[10][1]-50),scale=2)
        # d = 55
        # f = (w*d)/W
        #
    cv2.imshow("image",img)
    cv2.waitKey(1)