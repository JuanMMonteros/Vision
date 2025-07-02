import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

# Define constants
IMG_HEIGHT = 128  # Increased resolution for better feature learning
IMG_WIDTH = 128
BATCH_SIZE = 32

def create_model():
    """Builds and returns the CNN model."""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # Regularization to prevent overfitting
        layers.Dense(1, activation='sigmoid') # Sigmoid for binary classification
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(data_dir, epochs, model_path):
    """
    Loads data from directories, trains the model, and saves it.
    """
   def train_model(data_dir, epochs, model_path):
    """
    Loads data from directories, trains the model, and saves it.
    """
    # --- ADD THESE LINES FOR DEBUGGING ---
    import os
    abs_train_path = os.path.abspath(os.path.join(data_dir, 'train'))
    abs_val_path = os.path.abspath(os.path.join(data_dir, 'val'))
    abs_test_path = os.path.abspath(os.path.join(data_dir, 'test')) # For completeness if you were loading test data directly

    print(f"DEBUG: Absolute path for train_ds: {abs_train_path}")
    print(f"DEBUG: Does train_ds path exist? {os.path.exists(abs_train_path)}")
    print(f"DEBUG: Is train_ds path a directory? {os.path.isdir(abs_train_path)}")

    print(f"DEBUG: Absolute path for val_ds: {abs_val_path}")
    print(f"DEBUG: Does val_ds path exist? {os.path.exists(abs_val_path)}")
    print(f"DEBUG: Is val_ds path a directory? {os.path.isdir(abs_val_path)}")

    # --- END DEBUGGING LINES ---


    print(f"Loading training data from: {os.path.join(data_dir, 'train')}")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'train'),
        labels='inferred',
        label_mode='binary', # 'binary' for 0/1 labels
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        interpolation='nearest',
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    print(f"Loading validation data from: {os.path.join(data_dir, 'val')}")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'val'),
        labels='inferred',
        label_mode='binary',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        interpolation='nearest',
        batch_size=BATCH_SIZE,
        shuffle=False # No need to shuffle validation data
    )
    # Normalize pixel values (0-255 to 0-1)
    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Data augmentation (optional but recommended for better generalization)
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])

    # Apply data augmentation to the training dataset
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


    model = create_model()
    model.summary()

    print("\nStarting model training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # Save the trained model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()


def classify_images(model_path, image_paths):
    """
    Loads a pre-trained model and classifies new images.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print("Model loaded successfully.")

    class_names = ['gato', 'perro'] # Make sure this order matches your label mapping (0 for cat, 1 for dog)

    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"Warning: Image not found at {img_path}. Skipping.")
            continue

        print(f"\nClassifying image: {img_path}")
        img = tf.keras.utils.load_img(
            img_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch
        img_array = img_array / 255.0 # Normalize the image

        predictions = model.predict(img_array)
        score = predictions[0]

        # Since it's binary (sigmoid activation), score will be a single value
        if score > 0.5:
            predicted_class = class_names[1] # Dog
            confidence = score * 100
        else:
            predicted_class = class_names[0] # Cat
            confidence = (1 - score) * 100

        print(f"Prediction: This image is a {predicted_class} with {confidence:.2f}% confidence.")

        # Optional: Display the image with its prediction
        plt.imshow(img)
        plt.title(f"Prediction: {predicted_class} ({confidence:.2f}%)")
        plt.axis('off')
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Cat-Dog Classifier using TensorFlow/Keras.")
    parser.add_argument("--mode", required=True, choices=["train", "test"],
                        help="Mode of operation: 'train' to train a new model, 'test' to classify images.")
    parser.add_argument("--data_dir", type=str,
                        help="Path to the root directory of your dataset (e.g., 'dataset/'). Required for 'train' mode.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train the model. Required for 'train' mode.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to save/load the model (e.g., 'saved_model/perro_gato_classifier.keras').")
    parser.add_argument("--image_paths", nargs='+',
                        help="List of image paths to classify. Required for 'test' mode.")

    args = parser.parse_args()

    if args.mode == "train":
        if not args.data_dir:
            parser.error("--data_dir is required for 'train' mode.")
        train_model(args.data_dir, args.epochs, args.model_path)
    elif args.mode == "test":
        if not args.image_paths:
            parser.error("--image_paths is required for 'test' mode.")
        classify_images(args.model_path, args.image_paths)

if __name__ == "__main__":
    main()