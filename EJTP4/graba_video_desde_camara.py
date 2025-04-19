#! /usr/bin/env python
# -*- coding: utf-8 -*-

import cv2

fps = 25
anchoalto = (1280,720)
# Configuración para MP4 con H.264 (requiere FFmpeg)
fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter('mivideo.mp4', fourcc, fps, anchoalto)
#sino probar  'MJPG' -> .avi 

# Capturar video de cámara
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
