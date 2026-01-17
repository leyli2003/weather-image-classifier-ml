from cleanup import load_data
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
        ('rf', RandomForestClassifier(random_state=42))
    ])

    # Grid search parameters
    parameters = {
        'rf__n_estimators': [50, 100, 200, 300],
        'rf__max_depth': [10, 20, None, 30, 5, 100, 1000, 500],
        'rf__min_samples_split': [2, 5, 10, 20, 50, 100],
        'rf__min_samples_leaf': [1, 2, 4, 16, 32]
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

    #joblib.dump(best_model, 'randomforestGridsearch_weather_classifier.joblib')
    #print("\nPipeline saved as 'randomforestGridsearch_weather_classifier.joblib'")

if __name__ == "__main__":
    main()