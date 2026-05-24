# 🐹 Hamster Cámara — Virtual Emotion Cam

Convierte tu webcam en una **cámara virtual** que reemplaza tu cara con imágenes animadas según tus emociones y gestos en tiempo real. Perfecto para videollamadas, streams y reuniones.

---

## ¿Qué hace?

Detecta tu emoción o gesto y muestra la imagen correspondiente a través de una cámara virtual que puedes seleccionar en Zoom, Meet, Discord, OBS, etc.

| Emoción / Gesto | Imagen mostrada |
|---|---|
| 😊 Feliz | Happy |
| 😠 Enojado | Angry |
| 😢 Triste | Vety Sad |
| 😐 Neutral / Serio | Serio |
| 🤔 Dedo en barbilla | Pensando |
| 🤫 Dedo en boca | Silencio |
| 😘 Labios fruncidos | Kiss |
| 🤷 Dos palmas abiertas | Idkn |
| 🖕 Dedo medio | ¡Sorpresa! |

---

## Requisitos del sistema

- Python **3.10 – 3.12**
- macOS (probado en M-series) o Linux
- **OBS Studio** con el plugin Virtual Camera instalado
- Webcam funcional

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/maxvaldes33/facedetector.git
cd facedetector
```

### 2. Crear entorno virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install opencv-python numpy fer mediapipe pyvirtualcam
```

> **Nota:** `fer` instala TensorFlow Lite automáticamente. Si tienes conflictos de versiones, usa `pip install fer tensorflow`.

### 4. Descargar modelos de MediaPipe

Los archivos `.task` ya están incluidos en el repositorio:
- `face_landmarker.task`
- `gesture_recognizer.task`

Si necesitas descargarlos manualmente:

```bash
# Face Landmarker
wget -O face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# Gesture Recognizer
wget -O gesture_recognizer.task \
  https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
```

---

## Uso

### 1. Abrir OBS Studio y activar Virtual Camera

1. Abre OBS Studio
2. Haz clic en **"Start Virtual Camera"** (panel inferior derecho)
3. Deja OBS abierto en segundo plano

### 2. Ejecutar el programa

```bash
python3 virtual_cam.py
```

Verás en la terminal:
```
Camara virtual iniciada. Q para salir.
Selecciona 'OBS Virtual Camera' en tu app de videollamada.
Usando: /dev/...
```

### 3. Seleccionar la cámara virtual

En Zoom, Meet, Discord u otra app:
- Ve a Configuración → Cámara
- Selecciona **"OBS Virtual Camera"**

### 4. Salir

Presiona **`Q`** en la ventana de preview para cerrar.

---

## Gestos disponibles

| Gesto | Cómo hacerlo |
|---|---|
| **Pensando** | Acerca el dedo índice a tu barbilla |
| **Silencio** | Acerca el dedo índice a tus labios |
| **Idkn (no sé)** | Abre ambas palmas frente a la cámara |
| **Dedo medio** | Extiende solo el dedo medio |
| **Kiss** | Frunce los labios |

---

## Configuración avanzada

Los parámetros principales están al inicio de `virtual_cam.py`:

```python
SMOOTH_FRAMES   = 10    # Frames de suavizado (más = menos alucinaciones)
ENTER_THRESHOLD = 0.35  # Confianza mínima para activar una emoción
KEEP_THRESHOLD  = 0.25  # Confianza mínima para mantener la emoción actual
SAD_THRESHOLD   = 0.55  # Umbral específico para tristeza

# Umbrales individuales por emoción
EMOTION_THRESHOLD = {
    "silencio": 0.22,
    "kiss":     0.20,
    "pensando": 0.22,
    "sad":      0.55,
}
```

### Agregar tus propias imágenes

1. Coloca el archivo `.jpg` en la carpeta `Images/`
2. Agrega la entrada en `EMOTION_TO_IMAGE` dentro de `virtual_cam.py`

---

## Estructura del proyecto

```
facedetector/
├── virtual_cam.py          # Script principal
├── face_landmarker.task    # Modelo MediaPipe para landmarks faciales
├── gesture_recognizer.task # Modelo MediaPipe para gestos de mano
├── Images/
│   ├── Happy.jpg
│   ├── Angry.jpg
│   ├── Vety Sad.jpg
│   ├── Serio.jpg
│   ├── Pensando.jpg
│   ├── silencio.jpg
│   ├── kiss.jpg
│   ├── Idkn.jpg
│   └── Parar dedo de al medio.jpg
└── README.md
```

---

## Solución de problemas

**"No se pudo abrir la camara"**
→ Verifica que ninguna otra app esté usando la webcam.

**La cámara virtual no aparece en Zoom/Meet**
→ Asegúrate de que OBS tiene Virtual Camera activa antes de abrir la app de videollamada.

**Las emociones cambian demasiado rápido**
→ Aumenta `SMOOTH_FRAMES` a 15 o 20.

**Tristeza aparece cuando no debería**
→ Aumenta `SAD_THRESHOLD` a 0.65 o 0.70.

**Kiss / Silencio / Pensando no se detectan**
→ Baja su valor en `EMOTION_THRESHOLD` (ej: `"kiss": 0.15`).

---

## Tecnologías utilizadas

- [OpenCV](https://opencv.org/) — captura y procesamiento de video
- [FER](https://github.com/justinshenk/fer) — detección de emociones faciales
- [MediaPipe](https://mediapipe.dev/) — landmarks faciales y reconocimiento de gestos
- [pyvirtualcam](https://github.com/letmaik/pyvirtualcam) — cámara virtual

---

## Licencia

MIT — úsalo, modifícalo y compártelo libremente.
