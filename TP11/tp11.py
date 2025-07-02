#!/usr/bin/env python3
#training python .\tp11.py --mode train --data_dir .\dateset\train\ --model_path .\saved_model\imagen_classifier.keras --epochs 10
#test python .\tp11.py --mode test --model_path .\saved_model\imagen_classifier.keras --image_path .\dateset\test\casa1.jpeg
#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

############################
# PARÁMETROS GLOBALES
############################
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ["dog", "cat"]  # Solo dos clases
NUM_CLASSES = len(CLASS_NAMES)  # = 2
CONFIDENCE_THRESHOLD = 0.70     # 70 %

# ----------------------------------------------------------------------------
# MODELO
# ----------------------------------------------------------------------------

def build_model() -> tf.keras.Model:
    """Base MobileNetV2 congelada + clasificador lineal."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = models.Model(inputs, outputs, name="perro_gato_classifier")
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# ----------------------------------------------------------------------------
# ENTRENAMIENTO
# ----------------------------------------------------------------------------

def train(data_dir: str, epochs: int, out_path: str) -> None:
    """Entrena y guarda el modelo en `out_path`."""
    gen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        validation_split=0.2,
    )
    train_ds = gen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
    )
    val_ds = gen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
    )

    model = build_model()
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    print("CLASES:", train_ds.class_indices)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    print(f"\n✅  Modelo guardado en: {out_path}\n")

# ----------------------------------------------------------------------------
# PREDICCIÓN / INFERENCIA
# ----------------------------------------------------------------------------

def preprocess_cv2(bgr_img):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    array = np.expand_dims(resized, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(array), rgb


def predict(model_path: str, image_paths: List[str]) -> None:
    """Carga `model_path` y clasifica cada imagen."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"No encuentro el modelo '{model_path}'. ¿Lo entrenaste?")

    model = tf.keras.models.load_model(model_path)

    for path in image_paths:
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"⚠️ No pude leer '{path}'.")
            continue
        pre, rgb = preprocess_cv2(bgr)
        preds = model.predict(pre, verbose=0)[0]

        # Mostrar porcentajes por clase
        print(f"\nImagen: {path}")
        for name, score in zip(CLASS_NAMES, preds):
            print(f"  {name:>6}: {score:.2%}")

        conf = float(np.max(preds))
        pred_idx = int(np.argmax(preds))

        if conf < CONFIDENCE_THRESHOLD:
            pred_label = "desconocido"
        else:
            pred_label = CLASS_NAMES[pred_idx]

        # Mostrar imagen con etiqueta
        plt.figure(); plt.imshow(rgb); plt.axis("off")
        plt.title(pred_label)
        if pred_label == "desconocido":
            print(f"  ⚠️ Confianza {conf:.2f} < {CONFIDENCE_THRESHOLD}. Clasificado como DESCONOCIDO.")
    plt.show()

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TP 11 – Clasificador CNN Perro vs Gato")
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--data_dir", help="Carpeta raíz con imágenes (obligatorio en train)")
    parser.add_argument("--model_path", default="saved_model/perro_gato_classifier.keras")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--image_paths", nargs="*", help="Imágenes a clasificar (modo test)")
    args = parser.parse_args()


    if args.mode == "train":
        if not args.data_dir:
            parser.error("--data_dir es obligatorio en modo train")
        train(args.data_dir, args.epochs, args.model_path)

    if args.mode == "test":
        if not args.image_paths:
            parser.error("Debes indicar al menos una imagen con --image_paths en modo test")
        predict(args.model_path, args.image_paths)


if __name__ == "__main__":
    main()
