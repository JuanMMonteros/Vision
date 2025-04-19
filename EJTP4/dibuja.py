#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv

azul = (255, 0, 0); verde = (0, 255, 0); rojo = (0, 0, 255)

# Creamos una imagen RGB de color negro
img = np.zeros((512, 512, 3), np.uint8)

cv.line(img, (100, 100), (200, 100), azul, 5)
cv.imshow('frame', img); key = cv.waitKey(0)
cv.rectangle(img, (284, 50), (410, 178), verde, 1)
cv.imshow('frame', img); key = cv.waitKey(0)
cv.circle(img, (347, 113), 63, rojo, -1)
cv.imshow('frame', img); key = cv.waitKey(0)
cv.ellipse(img, (256, 256), (100, 150), 0, 0, 180, rojo, -1)
cv.imshow('frame', img); key = cv.waitKey(0)
pts = np.array([[90, 45], [215, 5], [225, 55], [205, 35]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv.polylines(img, [pts], True, (0, 255, 255))
pts = np.array([[422, 45], [297, 5], [287, 55], [307, 35]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv.polylines(img, [pts], True, (0, 255, 255))
cv.imshow('frame', img); key = cv.waitKey(0)
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img, 'CV2025', (10, 500), font, 4, (255, 255, 255), 2, cv.LINE_AA)

cv.imshow('frame', img)
key = cv.waitKey(0)
cv.destroyAllWindows()
