import cv2
import numpy as np
import sys
import time 


if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    print('Pasa un nombre de archivo como primer argumento')
    sys.exit(0)


img = cv2.imread(filename)  # Lee la imagen
if img is None:
    raise FileNotFoundError("Image not found")
original_img = img.copy() 

rectangle_select=False
drawing = False  # true if mouse is pressed
ix, iy = -1, -1
fx, fy = -1, -1

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing, rectangle_select
    if event == cv2.EVENT_LBUTTONDOWN and rectangle_select is False:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and rectangle_select is False:
        if drawing is True:
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)        
    elif event == cv2.EVENT_LBUTTONUP and rectangle_select is False:
        fx, fy = x, y
        rectangle_select = True
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)

cv2.namedWindow(str(filename))
cv2.setMouseCallback(str(filename), draw_rectangle)

while True:
    cv2.imshow(str(filename), img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('r'):
        rectangle_select = False
        drawing=False
        ix, iy = -1, -1
        fx, fy = -1, -1
        img = original_img.copy()
    elif k == ord('g'):
        if ix > 0 and iy > 0 and fx > 0 and fy > 0:
            new_filename = f"{int(time.time())}.png"
            if new_filename.strip():
                rangey = slice(iy, fy) if fy > iy else slice(fy, iy)
                rangex = slice(ix, fx) if fx > ix else slice(fx, ix)
                cropped_img = original_img[rangey, rangex]
                cv2.imwrite(new_filename, cropped_img)
                print(f"Imagen guardada como '{new_filename}'")
            else:
                print("Nombre de archivo no válido. No se guardó la imagen.")
            # Reiniciar variables
            ix, iy = -1, -1
            fx, fy = -1, -1
            rectangle_select = False
            drawing=False
            img = original_img.copy()
        else:
            print("No se ha seleccionado un rectángulo")
    elif k == ord('q'):
        print("Saliendo...")
        break

    elif k == 27:  # Escape key
        break

cv2.destroyAllWindows()
