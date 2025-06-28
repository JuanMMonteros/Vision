import cv2
import numpy as np
import sys
import time   

if len(sys.argv) > 2:
    filename = sys.argv[1]
    filenameafin= sys.argv[2]
else:
    print('Pasa los nombres de los archivos como argumento')
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
    global rectangle_select, drawing, ix, iy, fx, fy, img, original_img, affine_points, affine_mode, homography_points, homography_mode
    rectangle_select = False
    drawing = False
    ix, iy = -1, -1
    fx, fy = -1, -1
    affine_points = []
    affine_mode = False
    homography_mode = []
    homography_mode =  False
    img = original_img.copy()

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing, rectangle_select, flag
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
        if flag == False:
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

affine_points = []
affine_mode = False

def select_affine_points(event, x, y, flags, param):
    global affine_points, img, affine_mode
    if affine_mode and event == cv2.EVENT_LBUTTONDOWN:
        if len(affine_points) < 3:
            affine_points.append([x, y])
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(str(filename), img)

def insert_image_affine(base_img, dst_pts):
    global filenameafin
    insert_img = cv2.imread(filenameafin)
    if insert_img is None:
        print("No se pudo cargar la imagen a insertar.")
        return base_img
    h, w = insert_img.shape[:2]
    # Puntos fuente: esquinas de la imagen a insertar
    src_pts = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
    dst_pts = np.float32(dst_pts)
    # Calcular la matriz afín
    M = cv2.getAffineTransform(src_pts, dst_pts)
    # Transformar la imagen a insertar
    warped = cv2.warpAffine(insert_img, M, (base_img.shape[1], base_img.shape[0]))
    # Crear máscara de la imagen a insertar
    mask = np.zeros((insert_img.shape[0], insert_img.shape[1]), dtype=np.uint8)
    mask[:] = 255
    warped_mask = cv2.warpAffine(mask, M, (base_img.shape[1], base_img.shape[0]))
    # Combinar usando la máscara
    result = base_img.copy()
    for c in range(3):
        result[:, :, c] = np.where(warped_mask > 0, warped[:, :, c], base_img[:, :, c])
    return result

homography_points = []
homography_mode = False

def select_homography_points(event, x, y, flags, param):
    global homography_points, img
    if homography_mode and event == cv2.EVENT_LBUTTONDOWN:
        if len(homography_points) < 4:
            homography_points.append([x, y])
            cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
            cv2.imshow(str(filename), img)

def compute_homography(src_pts, dst_pts):
    src = np.float32(src_pts)
    dst = np.float32(dst_pts)
    H = cv2.getPerspectiveTransform(src, dst)
    return H

def rectify_image(base_img, src_pts, width=400, height=300):
    dst_pts = np.float32([[0,0], [width-1,0], [width-1,height-1], [0,height-1]])
    H = compute_homography(src_pts, dst_pts)
    rectified = cv2.warpPerspective(base_img, H, (width, height))
    return rectified

def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4,2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]      # Top-left
    ordered[2] = pts[np.argmax(s)]      # Bottom-right
    ordered[1] = pts[np.argmin(diff)]   # Top-right
    ordered[3] = pts[np.argmax(diff)]   # Bottom-left
    return ordered


def mouse_callback(event, x, y, flags, param):
    if affine_mode:
        select_affine_points(event, x, y, flags, param)
    elif homography_mode:
        select_homography_points(event, x, y, flags, param)
    else:
        draw_rectangle(event, x, y, flags, param)

cv2.namedWindow(str(filename))
cv2.setMouseCallback(str(filename), mouse_callback)
# Bucle principal
flag = False
while True:
    cv2.imshow(str(filename), img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('r'):    
        reset_program()
        flag = False
    elif k == ord('g'):
        if ix > 0 and iy > 0 and fx > 0 and fy > 0:
            new_filename = f"{int(time.time())}.png"
            rangey = slice(iy, fy) if fy > iy else slice(fy, iy)
            rangex = slice(ix, fx) if fx > ix else slice(fx, ix)
            cropped_img = original_img[rangey, rangex]
            cv2.imwrite(new_filename, cropped_img)
            print(f"Imagen guardada como '{new_filename}'")
            reset_program()
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
            reset_program()
        else:
            print("No se ha seleccionado un rectángulo")
    elif k == ord('a'):
        flag = True
        print("Selecciona 3 puntos no colineales con el mouse.")
        affine_points.clear()
        affine_mode = True
        img[:] = original_img.copy()
        while len(affine_points) < 3:
            cv2.imshow(str(filename), img)
            cv2.waitKey(1)
        affine_mode = False
        img[:] = original_img.copy()
        img_result = insert_image_affine(original_img, affine_points)
        img[:] = img_result
        original_img[:] = img_result
        print("Imagen insertada usando transformación afín.")
        reset_program()
    elif k == ord('h'):
        flag = True
        print("Selecciona 4 puntos no colineales con el mouse para rectificar.")
        homography_points.clear()
        homography_mode = True
        img[:] = original_img.copy()
        while len(homography_points) < 4:
            cv2.imshow(str(filename), img)
            cv2.waitKey(1)
        homography_points = order_points(homography_points)
        homography_mode = False
        img[:] = original_img.copy()  # Limpia los círculos antes de rectificar
        rectified = rectify_image(original_img, homography_points)
        cv2.imshow("Rectificada", rectified)
        # Si quieres guardar la imagen rectificada:
        cv2.imwrite(f"rectificada_{int(time.time())}.png", rectified)
        print("Imagen rectificada mostrada y guardada.")
        reset_program()
    elif k == ord('q'):
        print("Saliendo...")
        break
    elif k == 27:  # Escape key
        break

cv2.destroyAllWindows()