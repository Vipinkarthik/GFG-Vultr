import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

def extract_frames(video_path, label, output_dir, max_frames=500):
    cap = cv2.VideoCapture(video_path)
    count = 0
    label_dir = os.path.join(output_dir, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (160, 160))
        frame_filename = os.path.join(label_dir, f"{label}_{count}.jpg")
        cv2.imwrite(frame_filename, frame)
        count += 1
    cap.release()
    print(f"Extracted {count} frames from {video_path} for label '{label}'.")

extract_frames("harassment_video.mp4", label="harassment", output_dir="frames")
extract_frames("non_harassment_video.mp4", label="non_harassment", output_dir="frames")

def load_data(data_dir, target_size=(160, 160)):
    images = []
    labels = []
    for label in ["harassment", "non_harassment"]:
        label_dir = os.path.join(data_dir, label)
        for filename in os.listdir(label_dir):
            img_path = os.path.join(label_dir, filename)
            img = load_img(img_path, target_size=target_size)
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(1 if label == "non_harassment" else 0)

    return np.array(images), np.array(labels)

X, y = load_data("frames")
X = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def create_model(input_shape=(160, 160, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = create_model()
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

model.save("non_harassment_detector_model.h5")
print("Model saved as 'non_harassment_detector_model.h5'")
