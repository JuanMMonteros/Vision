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

rectangle_select = False
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
            img[:] = original_img.copy()
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP and rectangle_select is False:
        fx, fy = x, y
        rectangle_select = True
        drawing = False
        cv2.rectangle(img, (ix, iy), (fx, fy), (0, 255, 0), 2)

def apply_euclidean_transformation_with_border(cropped_img, angle, tx, ty):
    # Agregar un borde alrededor de la imagen seleccionada
    border_size = max(cropped_img.shape[0], cropped_img.shape[1]) // 2
    bordered_img = cv2.copyMakeBorder(
        cropped_img,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)  # Color negro para el borde
    )
    
    # Obtener las dimensiones de la imagen con borde
    rows, cols = bordered_img.shape[:2]
    # Calcular el centro de la imagen
    center = (cols // 2, rows // 2)
    # Crear la matriz de rotación
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    # Aplicar la rotación
    rotated_img = cv2.warpAffine(bordered_img, rotation_matrix, (cols, rows))
    # Crear la matriz de traslación
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    # Aplicar la traslación
    transformed_img = cv2.warpAffine(rotated_img, translation_matrix, (cols, rows))
    return transformed_img

# Modificar la tecla 'e' para usar la nueva función
cv2.namedWindow(str(filename))
cv2.setMouseCallback(str(filename), draw_rectangle)

while True:
    cv2.imshow(str(filename), img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('r'):
        rectangle_select = False
        drawing = False
        ix, iy = -1, -1
        fx, fy = -1, -1
        img = original_img.copy()
    elif k == ord('g'):
        if ix > 0 and iy > 0 and fx > 0 and fy > 0:
            new_filename = f"{int(time.time())}.png"
            rangey = slice(iy, fy) if fy > iy else slice(fy, iy)
            rangex = slice(ix, fx) if fx > ix else slice(fx, ix)
            cropped_img = original_img[rangey, rangex]
            cv2.imwrite(new_filename, cropped_img)
            print(f"Imagen guardada como '{new_filename}'")
        else:
            print("No se ha seleccionado un rectángulo")
    elif k == ord('e'):
        if ix > 0 and iy > 0 and fx > 0 and fy > 0:
            angle = float(input("Ingrese el ángulo de rotación (en grados): "))
            tx = int(input("Ingrese la traslación en x: "))
            ty = int(input("Ingrese la traslación en y: "))
            rangey = slice(iy, fy) if fy > iy else slice(fy, iy)
            rangex = slice(ix, fx) if fx > ix else slice(fx, ix)
            cropped_img = original_img[rangey, rangex]
            transformed_img = apply_euclidean_transformation_with_border(cropped_img, angle, tx, ty)
            new_filename = f"transformed_{int(time.time())}.png"
            cv2.imwrite(new_filename, transformed_img)
            print(f"Imagen transformada guardada como '{new_filename}'")
        else:
            print("No se ha seleccionado un rectángulo")
    elif k == ord('q'):
        print("Saliendo...")
        break
    elif k == 27:  # Escape key
        break

cv2.destroyAllWindows()