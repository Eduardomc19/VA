import cv2
import numpy as np

img = cv2.imread('amste.jpg')
_, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('original', img)
cv2.imshow('thereshold', th)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('original', gray)
cv2.imshow('thereshold', th)
cv2.waitKey(0)
cv2.destroyAllWindows()

_, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('binary inv thereshold', th)
cv2.waitKey(0)
cv2.destroyAllWindows()

_, th = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
cv2.imshow('trunc thereshold', th)
cv2.waitKey(0)
cv2.destroyAllWindows()

_, th = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
cv2.imshow('to zero thereshold', th)
cv2.waitKey(0)
cv2.destroyAllWindows()

_, th = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)
cv2.imshow('to zero inv thereshold', th)
cv2.waitKey(0)
cv2.destroyAllWindows()

th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow('mean thereshold', th)
cv2.waitKey(0)
cv2.destroyAllWindows()

th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow('adaptive threshold',th)
cv2.waitKey(0)
cv2.destroyAllWindows()

_, th = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('otsu threshold',th)
cv2.waitKey(0)
cv2.destroyAllWindows()