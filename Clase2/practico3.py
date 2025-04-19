import cv2
import sys

if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    print('Pasa un nombre de archivo como primer argumento')
    sys.exit(0)

cap = cv2.VideoCapture(filename)

# Obtener propiedades del video
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
delay = int(1000 / fps)

print(f"Propiedades del video: {width}x{height} a {fps} FPS")

# Configurar VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter("Salida12.mp4", fourcc, fps, (width,height))

while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.write(gray)
        cv2.imshow('Video procesado', gray)
        
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()