import cv2
import numpy as np
import matplotlib.pyplot as plt


def suma(img_1, img_2):
    print('suma')

    suma = cv2.addWeighted(cv2.imread(img_1), 0.3, cv2.imread(img_2), 0.7, 1)
    cv2.imshow('suma', suma)

    img_1 = cv2.imread(img_1)
    cv2.imshow('imagen 1', img_1)

    img_2 = cv2.imread(img_2)
    cv2.imshow('imagen 2', img_2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resta(img_1, img_2):
    print('resta')

    resta = cv2.subtract(cv2.imread(img_1), cv2.imread(img_2))
    cv2.imshow('resta', resta)

    img_1 = cv2.imread(img_1)
    cv2.imshow('imagen 1', img_1)

    img_2 = cv2.imread(img_2)
    cv2.imshow('imagen 2', img_2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def division(img_1, img_2):
    print('division')

    division = cv2.divide(cv2.imread(img_1), 0.5, cv2.imread(img_2), 0.5, 1)
    cv2.imshow('division', division)

    img_1 = cv2.imread(img_1)
    cv2.imshow('imagen 1', img_1)

    img_2 = cv2.imread(img_2)
    cv2.imshow('imagen 2', img_2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def multiplicacion(img_1, img_2):
    print('multiplicacion')

    multiplicacion = cv2.multiply(cv2.imread(img_1), 0.3, cv2.imread(img_2), 0.7, 1)
    cv2.imshow('multiplicacion', multiplicacion)

    img_1 = cv2.imread(img_1)
    cv2.imshow('imagen 1', img_1)

    img_2 = cv2.imread(img_2)
    cv2.imshow('imagen 2', img_2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def conjuncion(img_1, img_2):
    print('conjuncion')

    conjuncion = cv2.bitwise_and(cv2.imread(img_1), cv2.imread(img_2))
    cv2.imshow('conjuncion', conjuncion)

    img_1 = cv2.imread(img_1)
    cv2.imshow('imagen 1', img_1)

    img_2 = cv2.imread(img_2)
    cv2.imshow('imagen 2', img_2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def disyuncion(img_1, img_2):
    print('disyuncion')

    disyuncion = cv2.bitwise_or(cv2.imread(img_1), cv2.imread(img_2))
    cv2.imshow('disyuncion', disyuncion)

    img_1 = cv2.imread(img_1)
    cv2.imshow('imagen 1', img_1)

    img_2 = cv2.imread(img_2)
    cv2.imshow('imagen 2', img_2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def negacion(img_1):
    print('negacion')

    negacion = cv2.bitwise_not(cv2.imread(img_1))
    cv2.imshow('negacion', negacion)

    img_1 = cv2.imread(img_1)
    cv2.imshow('imagen 1', img_1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def traslacion(img_1):
    print('traslacion')
    img = cv2.imread(img_1, 0)
    x = np.float32([[1, 0, 100], [0, 1, 100]])
    col, row = img.shape

    traslacion = cv2.warpAffine(img, x, (col, row))
    cv2.imshow('traslacion', traslacion)

    img_1 = cv2.imread(img_1)
    cv2.imshow('imagen 1', img_1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def escalado(img_1):
    print('escalado')

    escalado = cv2.resize(cv2.imread(img_1), (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('escalado', escalado)

    img_1 = cv2.imread(img_1)
    cv2.imshow('imagen 1', img_1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rotacion(img_1):
    print('rotacion')
    img = cv2.imread(img_1, 0)
    row, col = img.shape
    x = cv2.getRotationMatrix2D((col / 2, row / 2), 180, 1)
    rotacion = cv2.warpAffine(img, x, (col, row))

    cv2.imshow('rotacion', rotacion)

    img_1 = cv2.imread(img_1)
    cv2.imshow('imagen 1', img_1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


img_1 = 'ntr.jpg'
img_2 = 'pst.jpg'
x = input('ingrese E: ')

if x == 'E' or x == 'e':
    suma(img_1, img_2)
    resta(img_1, img_2)
    division(img_1, img_2)
    multiplicacion(img_1, img_2)
    conjuncion(img_1, img_2)
    disyuncion(img_1, img_2)
    negacion(img_1)
    traslacion(img_1)
    escalado(img_1)
    rotacion(img_1)