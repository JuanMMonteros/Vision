import cv2
import numpy as np
import sys
import pathlib

if len(sys.argv) < 2:
    print("Uso: python medir.py <imagen> [lado_cm]")
    sys.exit(1)

img_path = pathlib.Path(sys.argv[1])
aruco_size_cm = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0

img = cv2.imread(str(img_path))
if img is None:
    raise FileNotFoundError(f"No se pudo abrir {img_path}")

# --- Selección del ROI ---
print("🖱️  Seleccioná un rectángulo que incluya el ArUco y la zona a medir.")
roi = cv2.selectROI("Seleccionar ROI", img, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Seleccionar ROI")
x, y, w, h = map(int, roi)
if w == 0 or h == 0:
    sys.exit("❌ ROI inválida.")
roi_img = img[y:y+h, x:x+w].copy()

# --- Detectar ArUco ---
DICT_NAMES = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_5X5_100", "DICT_6X6_250",
    "DICT_7X7_100", "DICT_ARUCO_ORIGINAL"
]

def find_aruco(gray):
    for name in DICT_NAMES:
        if hasattr(cv2.aruco, name):
            aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
            if ids is not None and len(ids):
                return corners, ids, name
    return None, None, None

gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
corners, ids, dic_name = find_aruco(gray_roi)

if ids is None:
    sys.exit("❌ No se detectó ningún ArUco en el ROI.")

print(f"✅ ArUco detectado (diccionario {dic_name}, id {int(ids[0])})")

# --- Homografía desde el ArUco ---
marker = corners[0][0]  # (4,2)
marker_dst_len = 200  # px arbitrario para definir escala
dst_marker = np.array([
    [0, 0],
    [marker_dst_len - 1, 0],
    [marker_dst_len - 1, marker_dst_len - 1],
    [0, marker_dst_len - 1]
], dtype=np.float32)

H = cv2.getPerspectiveTransform(marker, dst_marker)

# --- Aplicar homografía a todo el ROI sin recorte ---
h_roi, w_roi = roi_img.shape[:2]
roi_corners = np.array([[0, 0], [w_roi, 0], [w_roi, h_roi], [0, h_roi]], dtype=np.float32)
warped_corners = cv2.perspectiveTransform(roi_corners[None, :, :], H)[0]
x_min, y_min = np.floor(warped_corners.min(axis=0)).astype(int)
x_max, y_max = np.ceil(warped_corners.max(axis=0)).astype(int)
new_w = x_max - x_min
new_h = y_max - y_min

T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
H_corrected = T @ H
rectified_full = cv2.warpPerspective(roi_img, H_corrected, (new_w, new_h))

# --- Resize a 1270x720 manteniendo escala ---
final_w, final_h = 1270, 720
scale_x = final_w / new_w
scale_y = final_h / new_h
scale_factor = (scale_x + scale_y) / 2  # promedio para mantener proporciones

# Calculamos nueva escala en metros por píxel
px_per_cm = marker_dst_len / aruco_size_cm
m_per_px_before_resize = 1 / (px_per_cm * 100)  # metros/pixel antes del resize
m_per_px = m_per_px_before_resize / scale_factor  # ajustado al resize final

# Redimensionar imagen rectificada a 1270x720
rectified = cv2.resize(rectified_full, (final_w, final_h), interpolation=cv2.INTER_LINEAR)
display = rectified.copy()

print(f"📏 Escala final: {m_per_px:.6f} m/pix  ({1/(m_per_px*100):.2f} px/cm)")

# --- Medición interactiva ---
measure_pts = []

def mouse_measure(event, x, y, flags, param):
    global measure_pts, display
    if event == cv2.EVENT_LBUTTONDOWN:
        measure_pts.append((x, y))
        cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
        if len(measure_pts) == 2:
            p1, p2 = map(np.array, measure_pts)
            dist_px = np.linalg.norm(p1 - p2)
            dist_m = dist_px * m_per_px
            mid = tuple(((p1 + p2) / 2).astype(int))
            cv2.line(display, tuple(p1), tuple(p2), (255, 0, 0), 2)
            cv2.putText(display, f"{dist_m:.3f} m", mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print(f"📐 Distancia: {dist_m:.3f} m ({dist_px:.1f} px)")
            measure_pts.clear()
        cv2.imshow("Rectificada", display)

cv2.namedWindow("Rectificada")
cv2.setMouseCallback("Rectificada", mouse_measure)
print("🖱️  Medí con dos clics. 'r' reinicia, 'q' o 'ESC' para salir.")

while True:
    cv2.imshow("Rectificada", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        display = rectified.copy()
        measure_pts.clear()
        print("↺ Imagen restaurada.")
    elif key in (ord('q'), 27):
        break

cv2.destroyAllWindows()
