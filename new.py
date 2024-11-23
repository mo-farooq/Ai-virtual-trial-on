import os
import cvzone
import cv2
from cvzone.PoseModule import PoseDetector


# cap=cv2.VideoCapture("resources/Videos/video.mp4")
cap=cv2.VideoCapture(0)
detector=PoseDetector()

ShirtFolderPath="resources/Shirts"
ListShirts = [file for file in os.listdir(ShirtFolderPath) if file != '.DS_Store']

shirtProperties = {
    "1.png": {
        "ratios":(250/190,581/440),
        "offsets":(44,54),
    },
    "2.png": {
        "ratios":(250/190,581/440),
        "offsets":(44,54),
    },
    "3.png": {
        "ratios":(250/190,581/440),
        "offsets":(44,54),
    },
    "t1.png": {
        "ratios":(380/190, 600/500),
        "offsets":(100,55),
    },
    "t2.png": {
        "ratios":(320/190, 600/500),
        "offsets":(65,65),
    },
    "women.png": {
        "ratios":(320/190, 600/500),
        "offsets":(65,65),
    },
    "women2.png": {
        "ratios":(320/190, 600/500),
        "offsets":(65,65),
    },
"women3.png": {
        "ratios":(320/190, 600/500),
        "offsets":(65,65),
    },

}

print(ListShirts)
imageNo=2

imageButtonRight = cv2.imread("resources/button.png",cv2.IMREAD_UNCHANGED)
imageButtonLeft = cv2.flip(imageButtonRight,1)
counterRight=0
counterLeft=0
selectionSpeed = 10

while True:
         success,img = cap.read()
         img = cv2.flip(img, 1)
         img = detector.findPose(img,draw=False)
         lmList, bboxInfo = detector.findPosition(img, draw=False, bboxWithHands=False)

         if lmList:
             lm11=lmList[12][1:3]
             lm12=lmList[11][1:3]

             shirt_name=ListShirts[imageNo]
             imgShirt=cv2.imread(os.path.join(ShirtFolderPath,shirt_name),cv2.IMREAD_UNCHANGED)

             properties = shirtProperties.get(shirt_name,{"ratios":(1,1),"offsets":(0,0)})
             fixedRatio, shirtRatioHeightWidth = properties["ratios"]
             offsetx, offsety = properties["offsets"]

             widthofshirt= int((lm12[0]-lm11[0])*fixedRatio)
             imgShirt=cv2.resize(imgShirt,(widthofshirt,int(widthofshirt*shirtRatioHeightWidth)))

             print(widthofshirt)

             currentScale = (lm12[0]-lm11[0]) /190
             offset= int(offsetx*currentScale),int(offsety*currentScale)

             try:
                img=cvzone.overlayPNG(img,imgShirt,(lm11[0]-offset[0],lm11[1]-offset[1]))
             except:
                 pass
             img=cvzone.overlayPNG(img,imageButtonRight ,(1074,293))
             img=cvzone.overlayPNG(img,imageButtonLeft  ,(72,293))


             if  lmList[16][1]< 300:
                 counterRight+=1
                 cv2.ellipse(img,(139,360),(66,66),0,0,
                             counterRight*selectionSpeed,(0,255,0),20)
                 if counterRight*selectionSpeed>360:
                     counterRight=0
                     if imageNo< len(ListShirts)-1:
                       imageNo+=1


             elif lmList[15][1]>900:
                counterLeft += 1
                cv2.ellipse(img, (1138, 360), (66, 66), 0, 0,
                         counterLeft * selectionSpeed, (0, 255, 0), 20)
                if counterLeft * selectionSpeed > 360:
                 counterLeft = 0
                 if imageNo > 0:
                    imageNo -= 1
             else:
                 counterRight = 0
                 counterLeft = 0



         cv2.imshow("Image",img)
         cv2.waitKey(1)

