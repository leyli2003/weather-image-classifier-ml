import pandas as pd
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
import joblib


class ImagePreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, image_size=(64, 64)):
        self.image_size = image_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        processed_images = []

        for img in X:
            # Resize to fixed size
            img_resized = img.resize(self.image_size)

            # Convert to array
            img_array = np.array(img_resized)

            # Normalize to [0, 1]
            img_array = img_array / 255.0

            # Flatten
            img_flat = img_array.flatten()

            processed_images.append(img_flat)

        return np.array(processed_images)


def load_raw_data():

    # Paths
    images_folder = 'images'
    labels_file = 'labels.csv'

    df = pd.read_csv(labels_file)
    available_images = set(os.listdir(images_folder))

    # Filter to only available images
    df = df[df['filename'].isin(available_images)]

    X = []  # Images
    y = []  # Labels

    for idx, row in df.iterrows():
        # Load
        img_path = os.path.join(images_folder, row['filename'])
        img = Image.open(img_path)

        X.append(img)
        y.append(row['label'])

    # Manual label encoding
    unique_labels = sorted(np.unique(y))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}

    # Encode labels
    y_encoded = np.array([label_map[label] for label in y])

    print(f"Loaded {len(X)} raw images")
    print(f"Label encoding:")
    for label, idx in label_map.items():
        count = np.sum(y_encoded == idx)
        print(f"  {idx} - {label}: {count} images")

    return X, y_encoded


def main():
    # Load raw data
    print("Loading raw data...")
    X_raw, y = load_raw_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.1, random_state=42, stratify=y
    )

    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    # Preprocessing pipeline
    preprocessing_pipeline = Pipeline([
        ('image_preprocessor', ImagePreprocessor(image_size=(64, 64))),
        ('scaler', StandardScaler())
    ])

    # pipeline
    full_pipeline = Pipeline([
        ('preprocessor', preprocessing_pipeline),
        ('classifier', MLPClassifier(
            hidden_layer_sizes=(1000, 500, 250, 100),
            activation='relu',
            alpha=0.0001,
            max_iter=300,
            random_state=42,
            verbose=True
        ))
    ])

    # Train
    print("\nTraining model...")
    full_pipeline.fit(X_train, y_train)

    # Test
    print("\nEvaluating on test set...")
    y_pred = full_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save full pipeline
    joblib.dump(full_pipeline, 'mlp.joblib')
    print("\nFull pipeline saved as 'mlp_weather_classifier_full_pipeline.joblib'")

if __name__ == "__main__":
    main()