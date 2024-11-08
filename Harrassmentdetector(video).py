import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from tensorflow.keras.models import load_model

harassment_model = load_model('Harrassment Videos/harrassment_detection_model.h5')

def preprocess_frame(frame, target_size=(160, 160)):
    
    resized_frame = cv2.resize(frame, target_size)
    normalized_frame = resized_frame.astype('float32') / 255.0
    batch_frame = np.expand_dims(normalized_frame, axis=0)
    return batch_frame

def predict_harassment(video_path):
    cap = cv2.VideoCapture(video_path)
    harassment_detected = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = preprocess_frame(frame)
        prediction = harassment_model.predict(processed_frame)
        
        if prediction[0][1] > 0.5:
            harassment_detected = True
            break

    cap.release()
    return harassment_detected

def upload_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")])
    if not video_path:
        return

    status_var.set("Processing...")
    root.update_idletasks()

    harassment_detected = predict_harassment(video_path)

    if harassment_detected:
        result_var.set("Harassment activity detected in the video.")
    else:
        result_var.set("No harassment activity detected in the video.")

    status_var.set("Processing complete.")
    root.update_idletasks()

root = tk.Tk()
root.title("Harassment Detection")

header_frame = tk.Frame(root, pady=10)
header_frame.pack(fill=tk.X)

header_label = tk.Label(header_frame, text="Harassment Detection System", font=("Helvetica", 16, "bold"))
header_label.pack()

upload_btn = tk.Button(root, text="Upload Video", command=upload_video)
upload_btn.pack(pady=20)

result_frame = tk.Frame(root, padx=10, pady=10)
result_frame.pack(padx=20, pady=10, fill=tk.X)

result_var = tk.StringVar()
result_label = tk.Label(result_frame, textvariable=result_var, wraplength=400, font=("Helvetica", 12))
result_label.pack()

status_var = tk.StringVar()
status_label = tk.Label(root, textvariable=status_var, font=("Helvetica", 10, "italic"))
status_label.pack(pady=10)

root.mainloop()
