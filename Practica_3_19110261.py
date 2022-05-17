import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt  # carga la librería para graficar


control = 0
img1 = cv.imread('ntr.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('pst.jpg', cv.IMREAD_GRAYSCALE)
equ_img1 = cv.equalizeHist(img1)
equ_img2 = cv.equalizeHist(img2)

control = 15


def ecualizado(operacion, op_ecualizada):
    fig, ax = plt.subplots(2, 3)
    fig.set_size_inches(18.5, 10.5)

    cv.imshow('Imagen 1 eq', equ_img1)
    cv.imshow('Imagen 2 eq', equ_img2)
    cv.imshow('Imagen Ecualizada', op_ecualizada)
    cv.moveWindow('Imagen 1 eq', -1900, 700)
    cv.moveWindow('Imagen 2 eq', 1400, 700)
    cv.moveWindow('Imagen Ecualizada', 700, 1400)

    # Histograma no ecualizado
    ax[0, 0].hist(img1.ravel(), 256, [0, 256])
    ax[0, 0].set_title("Histograma Img1")

    ax[0, 1].hist(operacion.ravel(), 256, [0, 256])
    ax[0, 1].set_title("Histograma Operacion")

    ax[0, 2].hist(img2.ravel(), 256, [0, 256])
    ax[0, 2].set_title("Histograma Img2")

    # Histogramas ecualizados
    ax[1, 0].hist(equ_img1.ravel(), 256, [0, 256])
    ax[1, 0].set_title("Histograma Ecualizado Img1")

    ax[1, 1].hist(op_ecualizada.ravel(), 256, [0, 256])
    ax[1, 1].set_title("Histograma Ecualizado Operacion")

    ax[1, 2].hist(equ_img2.ravel(), 256, [0, 256])
    ax[1, 2].set_title("Histograma Ecualizado Img1")

    plt.show()


while control < 17:
    letra = input("Ingresa la J\n")
    if letra == "J":
        control = control + 1

    if control == 1:

        suma = cv.addWeighted(img1, 0.5, img2, 0.5, 1)
        equ_suma = cv.equalizeHist(suma)

        cv.imshow('Suma', suma)
        cv.imshow('Imagen 1', img1)
        cv.imshow('Imagen 2', img2)

        cv.moveWindow('Imagen 1', -1900, 0)
        cv.moveWindow('Imagen 2', 1400, 0)
        cv.moveWindow('Suma', 700, 0)

        print("Suma")
        # Función para generalizar el ecualizado
        ecualizado(suma, equ_suma)

        cv.waitKey(0)
        cv.destroyAllWindows()

    elif control == 2:
        print("Resta")
        # ax[0, 0].remove()
        # ax[0, 1].clear()
        # ax[0, 2].clear()
        # ax[1, 0].clear()
        # ax[1, 1].clear()
        # ax[1, 2].clear()

        resta = cv.subtract(img1, img2)
        equ_resta = cv.equalizeHist(resta)
        # print('img1[0,0]= ',img1[0,0])
        # print('img2[0,0]= ',img2[0,0])
        # print('resultado[0,0]= ',resultado[0,0])

        cv.imshow('Resta', resta)
        cv.imshow('Imagen 1', img1)
        cv.imshow('Imagen 2', img2)

        cv.moveWindow('Imagen 1', -1900, 0)
        cv.moveWindow('Imagen 2', 1400, 0)
        cv.moveWindow('Resta', 700, 0)

        ecualizado(resta, equ_resta)

        cv.waitKey(0)
        cv.destroyAllWindows()

    elif control == 3:
        print("Division")

        division = cv.divide(img1, img2)
        equ_division = cv.equalizeHist(division)

        cv.imshow('Division', division)
        cv.imshow('Imagen 1', img1)
        cv.imshow('Imagen 2', img2)

        cv.moveWindow('Imagen 1', -1900, 0)
        cv.moveWindow('Imagen 2', 1400, 0)
        cv.moveWindow('Division', 700, 0)

        ecualizado(division, equ_division)

        cv.waitKey(0)
        cv.destroyAllWindows()
    elif control == 4:
        print("Multiplicacion")

        multiplicacion = cv.multiply(img1, img2)
        equ_multiplicacion = cv.equalizeHist(multiplicacion)

        cv.imshow('Multiplicacion', multiplicacion)
        cv.imshow('Imagen 1', img1)
        cv.imshow('Imagen 2', img2)

        cv.moveWindow('Imagen 1', -1900, 0)
        cv.moveWindow('Imagen 2', 1400, 0)
        cv.moveWindow('Multiplicacion', 700, 0)

        ecualizado(multiplicacion, equ_multiplicacion)

        cv.waitKey(0)
        cv.destroyAllWindows()

    elif control == 5:
        print("Log Natural")

        img00 = np.uint8(np.log(img1))
        log_normalizado = cv.normalize(img00, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        equ_log = cv.equalizeHist(log_normalizado)

        cv.imshow('Log Natural', log_normalizado)
        cv.imshow('Imagen 1', img1)
        cv.imshow('Imagen 2', img2)

        cv.moveWindow('Imagen 1', -1900, 0)
        cv.moveWindow('Imagen 2', 1400, 0)
        cv.moveWindow('Log Natural', 700, 0)

        ecualizado(log_normalizado, equ_log)
        cv.waitKey(0)
        cv.destroyAllWindows()

    elif control == 6:
        print("Raiz")

        img00 = np.uint8(np.sqrt(img1))
        raiz_normalizada = cv.normalize(img00, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        equ_raiz = cv.equalizeHist(raiz_normalizada)

        cv.imshow('Raiz', raiz_normalizada)
        cv.imshow('Imagen 1', img1)
        cv.imshow('Imagen 2', img2)

        cv.moveWindow('Imagen 1', -1900, 0)
        cv.moveWindow('Imagen 2', 1400, 0)
        cv.moveWindow('Raiz', 700, 0)

        ecualizado(raiz_normalizada, equ_raiz)
        cv.waitKey(0)
        cv.destroyAllWindows()

    elif control == 7:
        print("Derivada")

        resultadox = cv.Sobel(img1, cv.CV_8U, 1, 0, ksize=3)  # Derivada en x
        resultadoy = cv.Sobel(img2, cv.CV_8U, 0, 1, ksize=3)  # Derivada en y
        imgs = cv.hconcat([resultadox, resultadoy])
        equ_derivadax = cv.equalizeHist(resultadox)

        cv.imshow('Derivada', imgs)
        cv.imshow('Imagen 1', img1)
        cv.imshow('Imagen 2', img2)

        cv.moveWindow('Imagen 1', -1900, 0)
        cv.moveWindow('Imagen 2', 1400, 0)
        cv.moveWindow('Derivada', 700, 0)

        ecualizado(resultadox, equ_derivadax)
        cv.waitKey(0)
        cv.destroyAllWindows()

    elif control == 8:
        print("Potencia")

        potencia = cv.pow(img2, 4)
        equ_potencia = cv.equalizeHist(potencia)

        cv.imshow('Potencia', potencia)
        cv.imshow('Imagen 1', img1)
        cv.imshow('Imagen 2', img2)

        cv.moveWindow('Imagen 1', -1900, 0)
        cv.moveWindow('Imagen 2', 1400, 0)
        cv.moveWindow('Potencia', 700, 0)

        ecualizado(potencia, equ_potencia)
        cv.waitKey(0)
        cv.destroyAllWindows()

    elif control == 9:
        print("Transpuesta")

        transpuesta = cv.transpose(img1)
        equ_transpuesta = cv.equalizeHist(transpuesta)

        cv.imshow('Transpuesta', transpuesta)
        cv.imshow('Imagen 1', img1)
        cv.imshow('Imagen 2', img2)

        cv.moveWindow('Imagen 1', -1900, 0)
        cv.moveWindow('Imagen 2', 1400, 0)
        cv.moveWindow('Transpuesta', 700, 0)

        ecualizado(transpuesta, equ_transpuesta)
        cv.waitKey(0)
        cv.destroyAllWindows()

    elif control == 10:
        print("Proyeccion")

        proyeccion = cv.reduce(img1, 0, cv.REDUCE_SUM, dtype=cv.CV_32F)
        # equ_proyeccion = cv.equalizeHist(proyeccion)

        cv.imshow('Proyeccion', proyeccion)
        cv.imshow('Imagen 1', img1)
        cv.imshow('Imagen 2', img2)

        cv.moveWindow('Imagen 1', -1900, 0)
        cv.moveWindow('Imagen 2', 1400, 0)
        cv.moveWindow('Proyeccion', 700, 0)

        # ecualizado(proyeccion,equ_proyeccion)
        cv.waitKey(0)
        cv.destroyAllWindows()


    elif control == 11:
        print("Conjuncion")

        conjuncion = cv.bitwise_and(img1, img2)
        equ_conjuncion = cv.equalizeHist(conjuncion)

        cv.imshow('Conjuncion', conjuncion)
        cv.imshow('Imagen 1', img1)
        cv.imshow('Imagen 2', img2)

        cv.moveWindow('Imagen 1', -1900, 0)
        cv.moveWindow('Imagen 2', 1400, 0)
        cv.moveWindow('Conjuncion', 700, 0)

        ecualizado(conjuncion, equ_conjuncion)
        cv.waitKey(0)
        cv.destroyAllWindows()
    elif control == 12:
        print("Disyuncion")

        disyuncion = cv.bitwise_or(img1, img2)
        equ_disyuncion = cv.equalizeHist(disyuncion)

        cv.imshow('Disyuncion', disyuncion)
        cv.imshow('Imagen 1', img1)
        cv.imshow('Imagen 2', img2)

        cv.moveWindow('Imagen 1', -1900, 0)
        cv.moveWindow('Imagen 2', 1400, 0)
        cv.moveWindow('Disyuncion', 700, 0)

        ecualizado(disyuncion, equ_disyuncion)
        cv.waitKey(0)
        cv.destroyAllWindows()

    elif control == 13:
        print("Negacion")

        negacion = cv.bitwise_not(img1)
        equ_negacion = cv.equalizeHist(negacion)

        cv.imshow('Negacion', negacion)
        cv.imshow('Imagen 1', img1)
        cv.imshow('Imagen 2', img2)

        cv.moveWindow('Imagen 1', -1900, 0)
        cv.moveWindow('Imagen 2', 1400, 0)
        cv.moveWindow('Negacion', 700, 0)

        ecualizado(negacion, equ_negacion)
        cv.waitKey(0)
        cv.destroyAllWindows()

    elif control == 14:
        print("Traslacion")

        rows, cols = img1.shape
        M = np.float32([[1, 0, 210], [0, 1, 20]])
        traslacion = cv.warpAffine(img1, M, (cols, rows))
        equ_traslacion = cv.equalizeHist(traslacion)

        cv.imshow('Traslacion', traslacion)
        cv.imshow('Imagen 1', img1)
        cv.imshow('Imagen 2', img2)

        cv.moveWindow('Imagen 1', -1900, 0)
        cv.moveWindow('Imagen 2', 1400, 0)
        cv.moveWindow('Traslacion', 700, 0)

        ecualizado(traslacion, equ_traslacion)
        cv.waitKey(0)
        cv.destroyAllWindows()

    elif control == 15:
        print("Escalado")

        escalado = cv.resize(img1, (0, 0), fx=0.5, fy=0.5)
        equ_escalado = cv.equalizeHist(escalado)

        cv.imshow('Escalado', escalado)
        cv.imshow('Imagen 1', img1)
        cv.imshow('Imagen 2', img2)

        cv.moveWindow('Imagen 1', -1900, 0)
        cv.moveWindow('Imagen 2', 1400, 0)
        cv.moveWindow('Escalado', 700, 0)

        ecualizado(escalado, equ_escalado)
        cv.waitKey(0)
        cv.destroyAllWindows()

    elif control == 16:
        print("Rotacion")

        rows, cols = img1.shape
        M = cv.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
        rotacion = cv.warpAffine(img1, M, (cols, rows))
        equ_rotacion = cv.equalizeHist(rotacion)

        cv.imshow('Rotacion', rotacion)
        cv.imshow('Imagen 1', img1)
        cv.imshow('Imagen 2', img2)

        cv.moveWindow('Imagen 1', -1900, 0)
        cv.moveWindow('Imagen 2', 1400, 0)
        cv.moveWindow('Rotacion', 700, 0)

        ecualizado(rotacion, equ_rotacion)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print("Traslacion A fin")
        img1 = cv.imread('ntr.jpg')
        rows, cols, ch = img1.shape

        pts1 = np.float32([[100, 400], [400, 100], [100, 100]])
        pts2 = np.float32([[50, 300], [400, 200], [80, 150]])
        M = cv.getAffineTransform(pts1, pts2)
        dst = cv.warpAffine(img1, M, (cols, rows))

        plt.subplot(121), plt.imshow(img1), plt.title('Input')
        plt.subplot(122), plt.imshow(dst), plt.title('Output')
        plt.show()