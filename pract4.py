import numpy as np
import cv2
#Dibujar en la imagen
img = np.zeros((800, 800, 3), np.uint8)
            #inicio  fin       color     anncho
cv2.line(img, (0,0), (799,799), (255,0,0), 5)
cv2.rectangle(img, (399,0), (799,399), (0,255,0) ,3)
cv2.circle(img,(199,599), 200, (0,0,255), -1)

pts = np.array([[100,50], [200,300], [700,200], [500,100]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.polylines(img, [pts], True, (120, 102, 102))

font = cv2.FONT_ITALIC
cv2.putText(img, 'Eduardo', (399,399), font, 4, (255,255,255), 2, cv2.LINE_8)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
img = cv2.imread('pst.jpg')
r = cv2.selectROI(img) #roi; region of interest
roi = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[1] + r[2])]
print(r)
cv2.imshow('roi', roi)
cv2.waitKey(0)
cv2.destroyAllWindows()