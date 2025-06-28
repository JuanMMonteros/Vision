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
def reset_program():
    global rectangle_select, drawing, ix, iy, fx, fy, img, original_img
    rectangle_select = False
    drawing = False
    ix, iy = -1, -1
    fx, fy = -1, -1
    img = original_img.copy()

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

def euclidean_transformation(cropped_img, angle, tx, ty,scale=1):
    # Agregar un borde alrededor de la imagen seleccionada
    border_size = max(img.shape[0], img.shape[1]) // 2
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
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    # Aplicar la rotación
    rotated_img = cv2.warpAffine(bordered_img, rotation_matrix, (cols, rows))
    # Crear la matriz de traslación
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    # Aplicar la traslación
    transformed_img = cv2.warpAffine(rotated_img, translation_matrix, (cols, rows))
    return transformed_img


cv2.namedWindow(str(filename))
cv2.setMouseCallback(str(filename), draw_rectangle)

while True:
    cv2.imshow(str(filename), img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('r'):
        reset_program()
    elif k == ord('s'):
        if ix > 0 and iy > 0 and fx > 0 and fy > 0:
            scale = (input("Ingrese el factor de escala: "))
            angle = (input("Ingrese el ángulo de rotación (en grados): "))
            tx = (input("Ingrese la traslación en x: "))
            ty = (input("Ingrese la traslación en y: "))
            rangey = slice(iy, fy) if fy > iy else slice(fy, iy)
            rangex = slice(ix, fx) if fx > ix else slice(fx, ix)
            try:
                scale = float(scale)
                angle = float(angle)
                tx = int(tx)
                ty = int(ty)
                cropped_img = original_img[rangey, rangex]
                transformed_img = euclidean_transformation(cropped_img, angle, tx, ty, scale)
                new_filename = f"euclidean_transformed_recorter_{int(time.time())}.png"
                cv2.imwrite(new_filename, transformed_img)
                print(f"Imagen transformada guardada como '{new_filename}'")
            except ValueError:
                print("Error: los valores de escala, ángulo, traslación deben ser numéricos")
                reset_program()
        else:
            print("No se ha seleccionado un rectángulo")
        reset_program()
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
        reset_program()
    elif k == ord('e'):
        if ix > 0 and iy > 0 and fx > 0 and fy > 0:
            angle = (input("Ingrese el ángulo de rotación (en grados): "))
            tx = (input("Ingrese la traslación en x: "))
            ty = (input("Ingrese la traslación en y: "))
            rangey = slice(iy, fy) if fy > iy else slice(fy, iy)
            rangex = slice(ix, fx) if fx > ix else slice(fx, ix)
            try:
                angle = float(angle)
                tx = int(tx)
                ty = int(ty)
                cropped_img = original_img[rangey, rangex]
                transformed_img = euclidean_transformation(cropped_img, angle, tx, ty)
                new_filename = f"euclidean_transformed_{int(time.time())}.png"
                cv2.imwrite(new_filename, transformed_img)
                print(f"Imagen transformada guardada como '{new_filename}'")
            except ValueError:
                print("Error: los valores de ángulo, traslación deben ser numéricos")
                reset_program()
                continue
        else:
            print("No se ha seleccionado un rectángulo")
        reset_program()
    elif k == ord('q'):
        print("Saliendo...")
        break
    elif k == 27:  # Escape key
        break

cv2.destroyAllWindows()
