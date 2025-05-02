#! /usr/bin/env python
# -*- coding: utf-8 -*-


import cv2

img = cv2.imread('homero.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Imagen', img)
k = cv2.waitKey(0)

if k == ord('g'):  #si se presiona la letra 'g' se guarda la imagen 
    print('Guardando imagen en escala de grises.')
    cv2.imwrite('homero_gris.png', img)
else:
    print('Imagen no guardada.')

cv2.destroyAllWindows()
