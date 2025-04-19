#! /usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

azul = (255, 0, 0); verde = (0, 255, 0); rojo = (0, 0, 255)
dibujando = False  # True si el botón está presionado
modo = True  # si True, rectángulo, sino línea, cambia con 'm'
xybutton_down = -1, -1

def dibuja(event, x, y, flags, param):
    global xybutton_down, dibujando, modo
    if event == cv2.EVENT_LBUTTONDOWN:
        dibujando = True
        xybutton_down = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if dibujando is True:
            img[:] = 0
            if modo is True:
                cv2.rectangle(img, xybutton_down, (x, y), azul, -1)
            else:
                cv2.line(img, xybutton_down, (x, y), rojo, 2)
    elif event == cv2.EVENT_LBUTTONUP:
        dibujando = False

img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', dibuja)
while(1):
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        modo = not modo
    elif k == 27:
        break
cv2.destroyAllWindows()
