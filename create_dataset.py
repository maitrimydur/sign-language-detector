import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Fix import for mp_drawing_styles
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the Hands solution
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Ensure the data directory exists (creates it if it doesn't)
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"Created directory {DATA_DIR}")
else:
    print(f"Directory {DATA_DIR} already exists")

data = []
labels = []

# Traverse all subdirectories in DATA_DIR
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)

    # Skip if the item is not a directory
    if not os.path.isdir(dir_path):
        continue

    # Process each image file within the subdirectory
    for img_path in os.listdir(dir_path):
        # Optionally skip non-image files by checking extensions
        if not (img_path.lower().endswith('.jpg') or 
                img_path.lower().endswith('.jpeg') or 
                img_path.lower().endswith('.png')):
            continue

        data_aux = []
        x_ = []
        y_ = []

        img_full_path = os.path.join(dir_path, img_path)
        img = cv2.imread(img_full_path)

        # Skip if the file is unreadable or not a valid image
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # First pass: collect all x, y coords to determine minimums
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                # Second pass: store normalized distances from minimum x, y
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

# Save the collected data using pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
