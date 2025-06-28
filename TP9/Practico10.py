import cv2
import numpy as np
import sys
import pathlib

# --- Parámetros ---
if len(sys.argv) < 2:
    print("Uso: python practico3.py <imagen> [lado_cm]")
    sys.exit(1)

img_path      = pathlib.Path(sys.argv[1])
aruco_size_cm = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0

img = cv2.imread(str(img_path))
if img is None:
    raise FileNotFoundError(f"No se pudo abrir '{img_path}'")

img_show = img.copy()

# --- Selección de 4 puntos para rectificar ---
rectify_points = []
selecting = True

def mouse_rectify(event, x, y, flags, param):
    global rectify_points, img_show, selecting
    if selecting and event == cv2.EVENT_LBUTTONDOWN:
        if len(rectify_points) < 4:
            rectify_points.append([x, y])
            cv2.circle(img_show, (x, y), 5, (0, 255, 255), -1)
            cv2.imshow("seleccion", img_show)

cv2.namedWindow("seleccion")
cv2.setMouseCallback("seleccion", mouse_rectify)
print("Selecciona 4 puntos no colineales para rectificar la imagen.")
while len(rectify_points) < 4:
    cv2.imshow("seleccion", img_show)
    if cv2.waitKey(1) & 0xFF == 27:
        sys.exit(0)
selecting = False

# --- Acomodar los puntos: TL, TR, BR, BL ---
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

rectify_points = order_points(rectify_points)

# --- Mostrar polígono sobre la imagen original para confirmar ---
img_confirm = img.copy()
pts = rectify_points.astype(int).reshape((-1,1,2))
cv2.polylines(img_confirm, [pts], isClosed=True, color=(0,255,255), thickness=2)
for idx, pt in enumerate(rectify_points):
    cv2.circle(img_confirm, tuple(pt.astype(int)), 7, (0,0,255), -1)
    cv2.putText(img_confirm, str(idx+1), tuple(pt.astype(int)+np.array([10,0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
cv2.imshow("seleccion", img_confirm)
print("Revisa el recuadro amarillo. Pulsa cualquier tecla para continuar o ESC para salir.")
key = cv2.waitKey(0)
if key == 27:
    sys.exit(0)
cv2.destroyWindow("seleccion")

# --- Rectificación ---
RECT_W, RECT_H = 1280, 720
dst_pts = np.float32([[0,0], [RECT_W-1,0], [RECT_W-1,RECT_H-1], [0,RECT_H-1]])
H = cv2.getPerspectiveTransform(rectify_points, dst_pts)
rect_img = cv2.warpPerspective(img, H, (RECT_W, RECT_H))

# --- Buscar ArUco en la imagen rectificada ---
DICT_NAMES = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL"
]

def find_aruco(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for name in DICT_NAMES:
        if not hasattr(cv2.aruco, name):
            continue
        aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
        if ids is not None and len(ids) > 0:
            return corners, ids, name
    return None, None, None

corners, ids, dic_found = find_aruco(rect_img)
if ids is None:
    print("⚠️  No se detectó ningún marcador ArUco en la imagen rectificada.")
    sys.exit(1)

print(f"👍 ArUco detectado: diccionario {dic_found}, id {int(ids[0])}")

# Usamos sólo el primer marcador encontrado
marker = corners[0][0]  # shape (4,2): TL, TR, BR, BL
cv2.polylines(rect_img, [marker.astype(int)], True, (0,255,0), 2)

# --- Escala real ---
side_px = int(np.linalg.norm(marker[1] - marker[0]))
px_per_cm = side_px / aruco_size_cm
cm_per_px = 1.0 / px_per_cm
print(f"Escala ≈ {cm_per_px:.5f} cm/px  ({px_per_cm:.2f} px/cm)")

# --- Medición interactiva ---
measure_pts = []
display = rect_img.copy()

def mouse_measure(event, x, y, flags, param):
    global measure_pts, display
    if event == cv2.EVENT_LBUTTONDOWN:
        measure_pts.append((x,y))
        cv2.circle(display, (x,y), 5, (0,0,255), -1)
        cv2.imshow("calibrada", display)
        if len(measure_pts) == 2:
            p1, p2 = map(np.array, measure_pts)
            dist_px = np.linalg.norm(p1-p2)
            dist_cm = dist_px * cm_per_px
            mid = tuple(((p1+p2)/2).astype(int))
            cv2.line(display, tuple(p1), tuple(p2), (255,0,0), 2)
            cv2.putText(display, f"{dist_cm:.2f} cm", mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
            cv2.imshow("calibrada", display)
            print(f"Distancia medida: {dist_cm:.2f} cm  ({dist_px:.1f} px)")
            measure_pts.clear()

print("💡 Ventana 'calibrada': clickea dos puntos para medir, 'r' reinicia, 'q' sale.")
cv2.namedWindow("calibrada")
cv2.setMouseCallback("calibrada", mouse_measure)

while True:
    cv2.imshow("calibrada", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        display = rect_img.copy()
        measure_pts.clear()
        print("↺ Ventana restaurada.")
    elif key in (ord('q'), 27):
        break

cv2.destroyAllWindows()