#! /usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

azul = (255, 0, 0); verde = (0, 255, 0); rojo = (0, 0, 255)
xybutton_down= 0,0
# mouse callback
def dibuja(event, x, y, flags, param):
    global xybutton_down
    if event == cv2.EVENT_LBUTTONDOWN:
        print("cv2.EVENT_LBUTTONDOWN", event)
        xybutton_down = x,y
        cv2.circle(img, xybutton_down, 9, rojo, 2)
    elif event == cv2.EVENT_RBUTTONDOWN:
        print("cv2.EVENT_RBUTTONDOWN", event)
        cv2.circle(img, (x, y), 5, verde, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        print("cv2.EVENT_LBUTTONUP", event)
        cv2.line(img, xybutton_down, (x, y), azul, 3)

img = np.ones((600, 800, 3), np.uint8) * 100
# Creamos una una ventana y capturamos los eventos del mouse en esa ventana
cv2.namedWindow('imagen')
cv2.setMouseCallback('imagen', dibuja)

while(1):
    # usamos la ventana creada para mostrar la imagen
    cv2.imshow('imagen', img)
    if cv2.waitKey(20) & 0xFF == 27: # ESC
        break
cv2.destroyAllWindows()
