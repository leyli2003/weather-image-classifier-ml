from cleanup import load_data
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

def main():
    # Load data
    print("Loading data...")
    X, y = load_data(image_size=(64, 64))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(max_iter=300, random_state=42))
    ])

    # Grid search parameters
    parameters = {
        'mlp__hidden_layer_sizes': [
            # 3 layers
            (150, 100, 50),
            (200, 150, 100),
            (300, 200, 100),
            (256, 128, 64),
            (512, 256, 128),

            # 4 layers
            (400, 300, 200, 100),
            (500, 400, 300, 200),
            (512, 256, 128, 64),
            (800, 400, 200, 100),
            (1000, 500, 250, 100),

            # 5 layers
            (500, 400, 300, 200, 100),
            (600, 500, 400, 300, 200),
            (800, 600, 400, 200, 100),
            (1000, 800, 600, 400, 200),

            # 6 layers
            (1000, 800, 600, 400, 200, 100),
            (512, 512, 256, 256, 128, 64)
        ],
        'mlp__activation': ['relu'],
        'mlp__alpha': [0.001]
    }

    # Grid search
    grid_search = GridSearchCV(
        pipeline,
        parameters,
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    print("\nStarting Grid Search...")
    grid_search.fit(X_train, y_train)
    print("Grid Search Complete!")

    print("\nBest parameters:")
    print(grid_search.best_params_)
    print(f"\nBest cross-validation score: {grid_search.best_score_:.4f}")

    # Best model
    best_model = grid_search.best_estimator_

    # Test
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    #joblib.dump(best_model, 'mlpGridSearch_weather_classifier.joblib')
    #print("\n Pipeline saved as 'mlpGridSearch_weather_classifier.joblib'")

if __name__ == "__main__":
    main()