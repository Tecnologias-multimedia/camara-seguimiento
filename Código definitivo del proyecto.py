#!/usr/bin/env python

# Importamos todas las librerías necesarias y las renombramos para poder trabajar más facilmente con ellas.
import numpy as np
import cv2.cv as cv
import os
from pantilthat import *

# Ejecutamos un comando en el sistema para cargar el modulo que carga el driver de Broadcom a Video4Linux.
os.system('sudo modprobe bcm2835-v4l2')

# Clasificador haar, carga la librería con el aprendizaje programado para reconocimiento facial de frente.
cascade = cv.Load('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')

# Definimos el movimiento inicial.
cam_pan = 0
cam_tilt = 0

# Movemos los servos a la posición inicial.
pan(-17)
tilt(0)

# Tamaño mínimo de la ventana con la que se va a hacer el reconocimiento, ponemos 20x20 ya que es el tamaño para el que se ha entrenado el clasificador.
tamVentana = (20, 20)

image_scale = 4

# El factor por el que la ventana de búsqueda se escala en los consiguientes escaneos.
factor_haar = 1.5

# Número mínimo (menos 1) de rectángulos adyacentes que componen un objeto.
# Se rechazan todos los grupos de un número menor de rectángulos que minAdyacentes-1.
# Si minAdyacentes es 0, la función devuelve todos los rectángulos candidatos detectados.
minAdyacentes = 2

# Tipo de operación. Actualmente, el único flag que se puede especificar es CV_HAAR_DO_CANNY_PRUNING.
# Con este flag, se utiliza el detector de bordes mediante el algoritmo de Canny para rechazar algunas regiones de imágenes que contienen muy pocos o demasiados bordes
# y, por lo tanto, no pueden contener el objeto buscado.
# Los valores del umbral se ajustan para la detección de rostros y en este caso la poda del árbol acelera el procesamiento.
haar_flag = cv.CV_HAAR_DO_CANNY_PRUNING

# Abrimos el stream de vídeo y creamos una ventana con la vista previa del mismo.
cap = cv.CreateCameraCapture(0)
cv.NamedWindow("Seguimiento", 1)

if cap:
    frame_copy = None

while(True):
    # Capturamos la imgen frame por frame y creamos la ventana en caso de que exista.
    frame = cv.QueryFrame(cap)
    if not frame:
        cv.WaitKey(0)
        break
    if not frame_copy:
        frame_copy = cv.CreateImage((frame.width,frame.height),
                                            cv.IPL_DEPTH_8U, frame.nChannels)


    # Reescalamos la imagen tanto en factores de alto como ancho a la hora de pasarla a grises y hacer el escalado posterior.
    gray = cv.CreateImage((frame.width,frame.height), 8, 1)
    small_img = cv.CreateImage((cv.Round(frame.width / image_scale),
                   cv.Round (frame.height / image_scale)), 8, 1)

    # Convertimos el color a escala de grises con la función BGR2GRAY para facilitar la detección ya que el color no nos es útil.
    cv.CvtColor(frame, gray, cv.CV_BGR2GRAY)

    # Reescalamos la imagen para aumentar el rendimiento.
    cv.Resize(gray, small_img, cv.CV_INTER_LINEAR)

    # Ecualizamos el histograma de los colores de la imagen en escala de grises para trabajar con valores de 0 a 255.
    cv.EqualizeHist(small_img, small_img)

    midFace = None

    if(cascade):
        t = cv.GetTickCount()
        # Utilizamos la función HaarDetectObjects con los factores que hemos definido, es decir, la cara de frente, la tolerancia, la imagen
        # pequeña reescalada en escala de grises, el tamaño de la ventana creciente y el mínimo de rectángulos adyacentes.
        faces = cv.HaarDetectObjects(small_img, cascade, cv.CreateMemStorage(0),
                                     factor_haar, minAdyacentes, haar_flag, tamVentana)
        t = cv.GetTickCount() - t
        if faces:


            for ((x, y, w, h), n) in faces:

            # Como la entrada en cv.HaarDetectObjects ha sido reescalada, reescalamos el recuadro de la cara y lo dibujamos con esas dimensiones
            # y el color RGB (100, 220, 255), que es azul claro.
                pt1 = (int(x * image_scale), int(y * image_scale))
                pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))
                cv.Rectangle(frame, pt1, pt2, cv.RGB(100, 220, 255), 1, 8, 0)
                # Obtener las cordenadas x e y de las cuatro esquinas del recuadro.
                x1 = pt1[0]
                x2 = pt2[0]
                y1 = pt1[1]
                y2 = pt2[1]
# Definimos que la cara esté justo en la mitad de la imagen.
                midFaceX = x1+((x2-x1)/2)
                midFaceY = y1+((y2-y1)/2)
                midFace = (midFaceX, midFaceY)
# Definimos la diferencia entre el centro de la imagen y donde está la cara.
                offsetX = midFaceX / float(frame.width/2)
                offsetY = midFaceY / float(frame.height/2)
                offsetX -= 1
                offsetY -= 1
# Definimos el movimiento anticipándonos a la velocidad de reacción y limitamos la salida al máximo y mínimo que nos da
# el pantilthat.
                cam_pan -= (offsetX * 5)
                cam_tilt += (offsetY * 5)
                cam_pan = max(-90,min(90,cam_pan))
                cam_tilt = max(-90,min(90,cam_tilt))
# Sacamos por consola los valores de diferencia, movimiento de los servos y tamaño de la imagen.
                print(offsetX, offsetY, midFace, cam_pan, cam_tilt, frame.width, frame.height)
# Movemos la cámara a las posiciones necesarias.
                pan(int(cam_pan))
                tilt(int(cam_tilt))
                break

    # Mostramos la imagen por pantalla y esperamos la pulsación de x para cerrar el programa.
    cv.ShowImage('Seguimiento',frame)
    if cv.WaitKey(1) & 0xFF == ord('x'):
        break

# Cerramos la ventana creada en caso de haber pulsado la tecla.
cv.DestroyWindow("Seguimiento")
