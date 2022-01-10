import cv2

img = cv2.imread("img/phone_aruco_marker.jpg")

cv2.imshow("Phone with Aruco", img)
cv2.waitKey(0)