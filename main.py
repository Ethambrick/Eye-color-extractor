# imports
from datetime import datetime
from data_loader import load_dataset
from image_processor import load_image, get_eye_crop, extract_color_distribution
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# setup
now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_file = open(f"results_{now}.txt", "w", encoding="utf-8", buffering=1)

def log(text):
    print(text)
    output_file.write(text + "\n")

def get_label(color_dist):
    return int(np.argmax(color_dist))

# load
data = load_dataset("archive/annotation/annotation")

log(f"Loaded samples: {len(data)}")
log(f"First sample: {data[0]}")

# build
X = []
y = []

for sample in data:
    file_name = sample["image"]

    image_path = None
    for folder in ["train_1", "train_2", "train_3", "train_4"]:
        path = os.path.join("archive", folder, folder, file_name)
        if os.path.exists(path):
            image_path = path
            break

    if image_path is None:
        continue

    image = load_image(image_path)
    eye_crop = get_eye_crop(image, sample["points"])

    if eye_crop is None or eye_crop.size == 0:
        continue

    eye_crop = cv2.resize(eye_crop, (64, 64))

    colors = extract_color_distribution(eye_crop)
    if sum(colors) == 0:
        continue

    label = get_label(colors)

    X.append(eye_crop)
    y.append(label)

log(f"Total samples used: {len(X)}")

# train
if len(X) == 0:
    log("ERROR: No data loaded. Check file paths.")
else:
    X = np.array(X)
    y = np.array(y)

    X_balanced = []
    y_balanced = []

    for i in range(len(X)):
        label = y[i]

        repeats = 1
        if label == 1:
            repeats = 10
        elif label == 2:
            repeats = 20

        for _ in range(repeats):
            X_balanced.append(X[i])
            y_balanced.append(y[i])

    X = np.array(X_balanced)
    y = np.array(y_balanced)

    X = X / 255.0
    y = to_categorical(y, num_classes=3)

    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    model = keras.Sequential([
        data_augmentation,

        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),

        layers.Dense(128, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    class_weights = {
        0: 1.0,
        1: 20.0,
        2: 40.0
    }

    history = model.fit(
        X, y,
        epochs=15,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weights
    )

    loss, acc = model.evaluate(X, y)
    log(f"Final Accuracy: {acc:.4f}")

    preds = model.predict(X)

    y_true = np.argmax(y, axis=1)
    y_pred = np.argmax(preds, axis=1)

    cm = confusion_matrix(y_true, y_pred, normalize='true')

    labels = ["Brown", "Blue", "Green"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Normalized Confusion Matrix")

    plt.savefig(f"confusion_matrix_heatmap_{now}.png")
    plt.close()

    log("Confusion Matrix:")
    log(str(cm))

    class_accuracy = np.diag(cm) / np.sum(cm, axis=1)

    plt.figure()
    plt.bar(labels, class_accuracy)

    plt.ylabel("Accuracy")
    plt.title("Accuracy per Class")

    plt.savefig(f"class_accuracy_{now}.png")
    plt.close()

# cleanup
output_file.close()