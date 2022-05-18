import cv2
import numpy as np
board = cv2.imread('emojis.jpg')
gray_board = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
queen = cv2.imread('cara1.jpg',0)
queen_w, queen_h = queen.shape[::-1]

match_1 = cv2.matchTemplate(gray_board, queen, cv2.TM_CCOEFF_NORMED)
threshold = 0.85

loc = np.where( match_1 >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(board, pt, (pt[0] + queen_w, pt[1] + queen_h), (0,0,255), 2)
pawn = cv2.imread('cara2.jpg',0)
pawn_w, pawn_h = pawn.shape[::-1]

match_1 = cv2.matchTemplate(gray_board, pawn, cv2.TM_CCOEFF_NORMED)
threshold = 0.85

loc = np.where( match_1 >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(board, pt, (pt[0] + pawn_w, pt[1] + pawn_h), (0, 0, 255), 2)
queen = cv2.imread('',0)
queen_w, queen_h = queen.shape[::-1]

match_1 = cv2.matchTemplate(gray_board, queen, cv2.TM_CCOEFF_NORMED)
threshold = 0.85

loc = np.where( match_1 >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(board, pt, (pt[0] + queen_w, pt[1] + queen_h), (0, 0, 255), 2)
cv2.imwrite('cara3.jpg', board)
True