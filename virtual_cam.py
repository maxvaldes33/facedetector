# IMPORTANTE: Seleccionar el intérprete correcto en VS Code antes de ejecutar.
# Cmd+Shift+P → "Python: Select Interpreter" → elegir el que tenga .pyenv o .venv
# Si no se selecciona el intérprete correcto, cv2/mediapipe/fer no van a importar.

import cv2
import numpy as np
import os
import time
import pyvirtualcam
from collections import deque
from fer.fer import FER
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "Images")
FACE_MODEL = os.path.join(BASE_DIR, "face_landmarker.task")
HAND_MODEL = os.path.join(BASE_DIR, "gesture_recognizer.task")

CAM_W, CAM_H = 1280, 720
SMOOTH_FRAMES        = 10     # más frames = menos alucinaciones
ENTER_THRESHOLD      = 0.35   # score mínimo para activar nueva emocion
KEEP_THRESHOLD       = 0.25   # score mínimo para mantener la emocion actual
CONFIDENCE_THRESHOLD = ENTER_THRESHOLD
SAD_THRESHOLD        = 0.55   # tristeza necesita score alto para activarse
# Thresholds individuales para gestos difíciles de activar
EMOTION_THRESHOLD = {
    "silencio": 0.22,
    "kiss":     0.20,
    "pensando": 0.22,
    "sad":      SAD_THRESHOLD,
}

# Emocion -> imagen a mostrar
EMOTION_TO_IMAGE = {
    "happy":        "Happy.jpg",
    "angry":        "Angry.jpg",
    "sad":          "Vety Sad.jpg",
    "Open_Palm":    "Idkn.jpg",
    "pensando":     "Pensando.jpg",
    "kiss":         "kiss.jpg",
    "silencio":     "silencio.jpg",
    "neutral":      "Serio.jpg",
    "middle_finger":"Parar dedo de al medio.jpg",
}

FER_EMOTIONS  = ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]
FACE_CUSTOM   = ["pensando", "kiss"]
HAND_GESTURES = ["None", "Closed_Fist", "Open_Palm", "Pointing_Up",
                 "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou",
                 "middle_finger", "pensando", "silencio"]
ALL_EMOTIONS  = FER_EMOTIONS + FACE_CUSTOM + HAND_GESTURES

# Prioridad: gestos de mano+cara primero, luego emociones FER
PRIORITY = ["middle_finger", "silencio", "pensando", "Open_Palm",
            "happy", "angry", "sad", "kiss", "neutral"]

# ---------------------------------------------------------------------------
# Face custom detectors (mismos que emotion_game.py)
# ---------------------------------------------------------------------------
LEFT_EYE_TOP  = 386; LEFT_EYE_BOT  = 374
RIGHT_EYE_TOP = 159; RIGHT_EYE_BOT = 145
MOUTH_TOP = 13; MOUTH_BOT = 14; MOUTH_LEFT = 61; MOUTH_RIGHT = 291

def lm_dist(lm, i, j):
    a = np.array([lm[i].x, lm[i].y])
    b = np.array([lm[j].x, lm[j].y])
    return float(np.linalg.norm(a - b))

def mouth_open(lm):
    h = lm_dist(lm, MOUTH_TOP, MOUTH_BOT)
    w = lm_dist(lm, MOUTH_LEFT, MOUTH_RIGHT)
    return float(np.clip((h / (w + 1e-6) - 0.05) / 0.3, 0, 1))

def brow_raise(lm):
    l = lm_dist(lm, 285, LEFT_EYE_TOP)
    r = lm_dist(lm, 55, RIGHT_EYE_TOP)
    eye_h = (lm_dist(lm, LEFT_EYE_TOP, LEFT_EYE_BOT) +
             lm_dist(lm, RIGHT_EYE_TOP, RIGHT_EYE_BOT)) / 2
    return float(np.clip(((l + r) / 2) / (eye_h + 1e-6) - 2.5, 0, 1) / 2.0)

def lip_pucker(lm):
    h  = lm_dist(lm, MOUTH_TOP, MOUTH_BOT)
    w  = lm_dist(lm, MOUTH_LEFT, MOUTH_RIGHT)
    fw = lm_dist(lm, 234, 454)
    return float(np.clip((0.38 - w / (fw + 1e-6)) / 0.12, 0, 1) *
                 np.clip((h / (w + 1e-6)) / 0.4, 0, 1))

def detect_face_custom(lm):
    pk = lip_pucker(lm)
    return {"kiss": float(np.clip(pk * 2.5, 0, 1))}

def detect_middle_finger(hand_lm):
    """Dedo medio extendido, índice + anular + meñique cerrados."""
    def ext(tip, pip): return hand_lm[tip].y < hand_lm[pip].y
    if ext(12, 10) and not ext(8, 6) and not ext(16, 14) and not ext(20, 18):
        return 1.0
    return 0.0

def detect_finger_near_face(hand_lm, face_lm, face_point_idx, threshold=0.10):
    """Detecta si el índice de la mano está cerca de un punto de la cara.
    Coordenadas normalizadas [0,1] — threshold en fracción del ancho."""
    tip = hand_lm[8]   # índice tip
    fp  = face_lm[face_point_idx]
    dist = ((tip.x - fp.x) ** 2 + (tip.y - fp.y) ** 2) ** 0.5
    score = float(np.clip(1.0 - dist / threshold, 0, 1))
    return score

CHIN_LANDMARK  = 152   # barbilla
MOUTH_LANDMARK = 13    # centro labio superior

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_and_resize(filename, w, h):
    path = os.path.join(IMAGES_DIR, filename)
    img = cv2.imread(path)
    if img is None:
        return None
    ih, iw = img.shape[:2]
    scale = min(h / ih, w / iw)
    resized = cv2.resize(img, (int(iw * scale), int(ih * scale)))
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    y0 = (h - resized.shape[0]) // 2
    x0 = (w - resized.shape[1]) // 2
    canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    return canvas

def draw_overlay(frame, emotion, score, transition):
    """Dibuja HUD discreto sobre el frame."""
    h, w = frame.shape[:2]
    # Barra inferior semitransparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    text = f"{emotion}  {int(score * 100)}%"
    cv2.putText(frame, text, (20, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 150), 2)
    # Barra de confianza
    bar_w = int((w - 40) * min(score / CONFIDENCE_THRESHOLD, 1.0))
    cv2.rectangle(frame, (20, h - 8), (20 + bar_w, h - 2), (0, 200, 100), -1)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    for path, name in [(FACE_MODEL, "face_landmarker.task"),
                       (HAND_MODEL, "gesture_recognizer.task")]:
        if not os.path.exists(path):
            print(f"Falta: {name}")
            return

    # Pre-cargar imágenes
    images = {}
    for emotion, filename in EMOTION_TO_IMAGE.items():
        img = load_and_resize(filename, CAM_W, CAM_H)
        if img is not None:
            images[emotion] = img
        else:
            print(f"No se pudo cargar {filename}")

    # Detectores
    fer_detector = FER(mtcnn=False)

    face_landmarker = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        ))

    gesture_recognizer = mp_vision.GestureRecognizer.create_from_options(
        mp_vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        ))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la camara.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    smooth_buffer = {e: deque(maxlen=SMOOTH_FRAMES) for e in ALL_EMOTIONS}
    frame_ts        = 0
    current_emotion = None
    current_img     = None


    print("Camara virtual iniciada. Q para salir.")
    print("Selecciona 'OBS Virtual Camera' en tu app de videollamada.")

    with pyvirtualcam.Camera(width=CAM_W, height=CAM_H, fps=30) as vcam:
        print(f"Usando: {vcam.device}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame    = cv2.flip(frame, 1)
            frame_ts += 33
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            raw = {e: 0.0 for e in ALL_EMOTIONS}

            # FER
            fer_res = fer_detector.detect_emotions(frame)
            if fer_res:
                for e, v in fer_res[0]["emotions"].items():
                    raw[e] = v

            # FaceLandmarker
            fl_res = face_landmarker.detect_for_video(mp_img, frame_ts)
            face_lm = fl_res.face_landmarks[0] if fl_res.face_landmarks else None
            if face_lm:
                custom = detect_face_custom(face_lm)
                raw["kiss"] = custom["kiss"]

            # GestureRecognizer + gestos custom con cara
            gr_res = gesture_recognizer.recognize_for_video(mp_img, frame_ts)
            open_palms = 0  # contar manos abiertas para idkn
            if gr_res.gestures:
                for i, gestures_per_hand in enumerate(gr_res.gestures):
                    top = gestures_per_hand[0]
                    if top.category_name in raw:
                        raw[top.category_name] = max(raw[top.category_name], top.score)
                    if top.category_name == "Open_Palm" and top.score > 0.5:
                        open_palms += 1
                    if gr_res.hand_landmarks and i < len(gr_res.hand_landmarks):
                        hand_lm = gr_res.hand_landmarks[i]
                        # Dedo medio
                        raw["middle_finger"] = max(raw["middle_finger"],
                                                   detect_middle_finger(hand_lm))
                        # Gestos que requieren cara + mano
                        if face_lm:
                            # Pensando: índice cerca de la barbilla — zona más amplia
                            raw["pensando"] = max(raw["pensando"],
                                detect_finger_near_face(hand_lm, face_lm, CHIN_LANDMARK, threshold=0.18))
                            # Silencio: índice cerca de la boca — zona más amplia
                            raw["silencio"] = max(raw["silencio"],
                                detect_finger_near_face(hand_lm, face_lm, MOUTH_LANDMARK, threshold=0.16))
            # Idkn: dos manos abiertas = encogerse de hombros
            raw["Open_Palm"] = 1.0 if open_palms >= 2 else 0.0

            # Suavizado
            for e in ALL_EMOTIONS:
                smooth_buffer[e].append(raw[e])
            emo_dict = {e: float(np.mean(smooth_buffer[e])) for e in ALL_EMOTIONS}

            # Histeresis: facil entrar, dificil salir accidentalmente
            current_score = emo_dict.get(current_emotion, 0.0) if current_emotion else 0.0

            if current_emotion and current_score < KEEP_THRESHOLD:
                # La emocion actual cayo — buscar nueva
                current_emotion = None
                current_img     = None

            if current_emotion is None:
                # Buscar nueva emocion por prioridad
                best_emotion = None
                best_score   = ENTER_THRESHOLD
                for e in PRIORITY:
                    s = emo_dict.get(e, 0.0)
                    # emociones con threshold propio ignoran best_score como mínimo
                    min_enter = EMOTION_THRESHOLD.get(e, best_score)
                    if s > min_enter and e in images:
                        best_score   = s
                        best_emotion = e
                # Fallback: serio es la imagen por defecto cuando no hay emocion
                if best_emotion is None:
                    best_emotion = "neutral"
                current_emotion = best_emotion
                current_img     = images[best_emotion].copy()
            else:
                # Permitir cambio directo si otra emocion supera a la actual por margen
                for e in PRIORITY:
                    if e == current_emotion:
                        continue
                    s = emo_dict.get(e, 0.0)
                    min_score = EMOTION_THRESHOLD.get(e, ENTER_THRESHOLD)
                    if s > current_score + 0.18 and s >= min_score and e in images:
                        current_emotion = e
                        current_img     = images[e].copy()
                        break

            # Frame de salida — serio es el default
            output = current_img.copy() if current_img is not None else images["neutral"].copy()

            lbl  = current_emotion or "ninguna"
            scr  = emo_dict.get(current_emotion, 0.0) if current_emotion else 0.0
            draw_overlay(output, lbl, scr, 0)

            # Enviar a camara virtual (convertir BGR→RGB)
            vcam.send(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            vcam.sleep_until_next_frame()

            # Preview local (opcional)
            small = cv2.resize(output, (640, 360))
            cv2.imshow("Virtual Cam Preview (Q para salir)", small)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    face_landmarker.close()
    gesture_recognizer.close()
    cv2.destroyAllWindows()
    print("Camara virtual cerrada.")


if __name__ == "__main__":
    main()
