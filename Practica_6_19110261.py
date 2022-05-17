import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while (1):
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    ##low_color = np.array([60, 35, 140]) #azul
    ##up_color = np.array([180, 255, 255])

    low_color = np.array([36, 0, 0])
    up_color = np.array([86, 255, 255])

    mask = cv2.inRange(hsv, low_color, up_color)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()