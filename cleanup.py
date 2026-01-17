import pandas as pd
import os
from PIL import Image
import numpy as np
def load_data(image_size=(128, 128)):

    # Paths
    images_folder = 'images'
    labels_file = 'labels.csv'

    df = pd.read_csv(labels_file)
    available_images = set(os.listdir(images_folder))

    # Filter to only available images
    df = df[df['filename'].isin(available_images)]
    print(f"Loading {len(df)} images...")
    print(f"Resizing all images to {image_size}...")

    X = []  # Images
    y = []  # Labels
    for idx, row in df.iterrows():
            # Load
            img_path = os.path.join(images_folder, row['filename'])
            img = Image.open(img_path)
            # Resize
            img = img.resize(image_size)
            # flatten
            img_array = np.array(img)
            img_array = img_array / 255.0  # Scale to [0, 1]
            img_flat = img_array.flatten()
            X.append(img_flat)
            y.append(row['label'])
    X = np.array(X)
    y = np.array(y)

    unique_labels = sorted(np.unique(y))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}

    y_encoded = np.array([label_map[label] for label in y])
    print(f"Loaded {len(X)} images")
    print(f"Image shape: {X.shape}")
    print(f" Label encoding:")
    for label, idx in label_map.items():
        count = np.sum(y_encoded == idx)
        print(f"  {idx} - {label}: {count} images")
    return X, y_encoded

if __name__ == "__main__":
    X, y = load_data()
    print("\nData ready for training!")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"y values: {np.unique(y)}")