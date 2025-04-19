import cv2
import numpy as np

def aplicar_umbral_manual(imagen, umbral):

    resultado = np.zeros_like(imagen)
    for y in range(imagen.shape[0]):
        for x in range(imagen.shape[1]):
            resultado[y, x] = 255 if imagen[y, x] > umbral else 0
    return resultado

# Cargar imagen en escala de grises (blanco y negro)
imagen = cv2.imread('hoja.png', cv2.IMREAD_GRAYSCALE)

# Verificar que la imagen se cargó correctamente
if imagen is None:
    print("Error al cargar la imagen")
    exit()

# Aplicar umbral (ajusta este valor según necesites, entre 0-255)
umbral = 128
imagen_umbralizada = aplicar_umbral_manual(imagen, umbral)
cv2.imshow("ventana",imagen_umbralizada)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Guardar el resultado
cv2.imwrite('imagen_umbralizada.jpg', imagen_umbralizada)

print("Proceso completado! Imagen guardada como 'imagen_umbralizada.jpg'")
