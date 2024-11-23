import cvzone
import cv2
import os
from cvzone.PoseModule import PoseDetector

# Initialize Pose Detector
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

detector = PoseDetector()

# Shirt folder and file paths
ShirtFolderPath = "resources/Shirts"
ListShirts = [file for file in os.listdir(ShirtFolderPath) if file.endswith('.png')]

if not ListShirts:
    print("Error: No shirts found in the folder.")
    exit()

# Shirt properties
shirtProperties = {
    "1.png": {"ratios": (250 / 190, 581 / 440), "offsets": (44, 54)},
    "2.png": {"ratios": (250 / 190, 581 / 440), "offsets": (44, 54)},
    "3.png": {"ratios": (250 / 190, 581 / 440), "offsets": (44, 54)},
    "t1.png": {"ratios": (380 / 190, 600 / 500), "offsets": (100, 55)},
    "t2.png": {"ratios": (320 / 190, 600 / 500), "offsets": (65, 65)},
    "women.png": {"ratios": (320 / 190, 600 / 500), "offsets": (65, 65)},
    "women2.png": {"ratios": (320 / 190, 600 / 500), "offsets": (65, 65)},
    "women3.png": {"ratios": (320 / 190, 600 / 500), "offsets": (65, 65)},
}

# Initial setup
currentSection = "Men"  # Default section
imageNo = 0
imageButtonRight = cv2.imread("resources/button.png", cv2.IMREAD_UNCHANGED)
imageButtonLeft = cv2.flip(imageButtonRight, 1)
counterRight = 0
counterLeft = 0
selectionSpeed = 10

if imageButtonRight is None:
    print("Error: Button image not found.")
    exit()

def filter_shirts(section):
    if section == "Men":
        return [s for s in ListShirts if "women" not in s]
    elif section == "Women":
        return [s for s in ListShirts if "women" in s]

while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to capture frame.")
        break

    img = cv2.flip(img, 1)
    img = detector.findPose(img, draw=False)
    lmList, bboxInfo = detector.findPosition(img, draw=False, bboxWithHands=False)

    if lmList:
        lm11 = lmList[12][1:3]
        lm12 = lmList[11][1:3]

        # Check hand positions to switch sections
        if lmList[15][1] < 200:  # Left hand raised
            currentSection = "Men"
            imageNo = 0  # Reset to first item in the section
        elif lmList[16][1] < 200:  # Right hand raised
            currentSection = "Women"
            imageNo = 0  # Reset to first item in the section

        # Get the filtered shirt list
        filteredShirts = filter_shirts(currentSection)
        if not filteredShirts:
            print(f"No shirts available in {currentSection} section.")
            continue

        shirt_name = filteredShirts[imageNo]
        shirt_path = os.path.join(ShirtFolderPath, shirt_name)
        if not os.path.exists(shirt_path):
            print(f"Shirt file not found: {shirt_path}")
            continue

        imgShirt = cv2.imread(shirt_path, cv2.IMREAD_UNCHANGED)

        # Get shirt properties
        properties = shirtProperties.get(shirt_name, {"ratios": (1, 1), "offsets": (0, 0)})
        fixedRatio, shirtRatioHeightWidth = properties["ratios"]
        offsetx, offsety = properties["offsets"]

        # Resize shirt
        widthofshirt = int((lm12[0] - lm11[0]) * fixedRatio)
        imgShirt = cv2.resize(imgShirt, (widthofshirt, int(widthofshirt * shirtRatioHeightWidth)))

        currentScale = (lm12[0] - lm11[0]) / 190
        offset = int(offsetx * currentScale), int(offsety * currentScale)

        try:
            img = cvzone.overlayPNG(img, imgShirt, (lm11[0] - offset[0], lm11[1] - offset[1]))
        except Exception as e:
            print(f"Error overlaying shirt: {e}")
            continue

        # Overlay navigation buttons
        img = cvzone.overlayPNG(img, imageButtonRight, (1074, 293))
        img = cvzone.overlayPNG(img, imageButtonLeft, (72, 293))

        # Navigation logic
        if lmList[16][1] < 300:  # Right hand near button
            counterRight += 1
            cv2.ellipse(img, (1138, 360), (66, 66), 0, 0, counterRight * selectionSpeed, (0, 255, 0), 20)
            if counterRight * selectionSpeed > 360:
                counterRight = 0
                if imageNo < len(filteredShirts) - 1:
                    imageNo += 1

        elif lmList[15][1] < 300:  # Left hand near button
            counterLeft += 1
            cv2.ellipse(img, (139, 360), (66, 66), 0, 0, counterLeft * selectionSpeed, (0, 255, 0), 20)
            if counterLeft * selectionSpeed > 360:
                counterLeft = 0
                if imageNo > 0:
                    imageNo -= 1
        else:
            counterRight = 0
            counterLeft = 0

        # Display the current section
        cv2.putText(img, f"Section: {currentSection}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key
        break

cap.release()
cv2.destroyAllWindows()