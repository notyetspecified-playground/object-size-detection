import cv2
import numpy as np

# load image
img = cv2.imread("img/test.jpg")

# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# create mask
mask = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5
)
# find contours
objects_contours, _ = cv2.findContours(
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# cleanup artifacts
contours = []
for cnt in objects_contours:
    area = cv2.contourArea(cnt)
    if area > 100:
        cnt = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        contours.append(cnt)

# aruco
parameters = cv2.aruco.DetectorParameters_create()
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
corners, ids, _ = cv2.aruco.detectMarkers(
    img, aruco_dict, parameters=parameters)
int_corners = np.int0(corners)
cv2.polylines(img, int_corners, True, (0, 255, 0), 5)
aruco_perimeter = cv2.arcLength(corners[0], True)
pixel_cm_ratio = aruco_perimeter / 12.25

# draw contours
for cnt in contours:
    rect = cv2.minAreaRect(cnt)
    (x, y), (w, h), angle = rect
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.polylines(img, [box], True, (255, 0, 0), 2)
    object_width = w / pixel_cm_ratio
    object_height = h / pixel_cm_ratio
    text = "{} x {}".format(round(object_height, 1), round(object_width, 1))
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1
    font_thickness = 1
    x = int(x)
    y = int(y)
    (text_w, text_h), _ = cv2.getTextSize(
        text, font, font_scale, font_thickness)
    y += int(text_h/2)
    x -= int(text_w/2)
    margin = 2
    cv2.rectangle(img, (x-margin, y+margin), (x + text_w +
                  margin, y - text_h-margin), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, font_scale,
                (100, 200, 0), font_thickness)

cv2.imshow("test", img)
cv2.waitKey(0)
