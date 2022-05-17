import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while (1):
    _, frame = cap.read()

    laplace = cv2.Laplacian(frame, cv2.CV_64F)
    sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
    border = cv2.Canny(frame, 200, 150)

    cv2.imshow('img', frame)
    cv2.imshow('laplaciano', laplace)
    cv2.imshow('sobel x', sobelx)
    cv2.imshow('sobel y', sobely)
    cv2.imshow('canny', border)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()