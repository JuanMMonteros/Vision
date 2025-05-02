import cv2
import sys

# Verificar argumentos de línea de comandos
if len(sys.argv) > 1:
    file_output = sys.argv[0]

else:
    print("Error: Debes pasar el nombre del archivo de salida como argumento")
    print("Ejemplo: python script.py Source Video.mp4")
    sys.exit(0)

# Inicializar captura de video (cámara web por defecto)
cap = cv2.VideoCapture(0)

# Configurar el video writer (ajusta el tamaño según tu cámara)
fourcc = cv2.VideoWriter_fourcc(*'H264')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter("Salida.mp4", fourcc, fps, (frame_width, frame_height))

print("Grabando... Presiona cualquier tecla para detener y guardar")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el frame")
        break
    
    # Mostrar el frame
    cv2.imshow('Grabando - Presiona cualquier tecla para detener', frame)
    
    # Guardar el frame en el video
    out.write(frame)
    
    # Salir si se presiona cualquier tecla
    if cv2.waitKey(1) != -1:
        break

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video guardado como {file_output}")


