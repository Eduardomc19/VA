import cv2
import numpy as np

frame = cv2.imread('paisaje.jpg')
hvs = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

low = np.array([60, 35, 140])
up = np.array([180, 255, 255])

mask = cv2.inRange(hvs, low, up)
res = cv2.bitwise_and(frame, frame, mask=mask)

kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(res, kernel, iterations=1)

dilation = cv2.dilate(res, kernel, iterations=1)

fp = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)

fn = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)

cv2.imshow('img', frame)
cv2.imshow('noise', res)
cv2.imshow('erosion', erosion)
cv2.imshow('dilation', dilation)
cv2.imshow('open', fp)
cv2.imshow('close', fn)

cv2.waitKey(0)
cv2.destroyAllWindows()
cap = cv2.VideoCapture(0)

while (1):
    _, frame = cap.read()
    hvs = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    low = np.array([60, 35, 140])
    up = np.array([180, 255, 255])

    mask = cv2.inRange(hvs, low, up)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(res, kernel, iterations=1)

    dilation = cv2.dilate(res, kernel, iterations=1)

    fp = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)

    fn = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('img', frame)
    cv2.imshow('erosion', erosion)
    cv2.imshow('dilation', dilation)
    cv2.imshow('open', fp)
    cv2.imshow('close', fn)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()