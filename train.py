from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "iris.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = BASE_DIR / "model" / "iris_model.pkl"

df = pd.read_csv(DATA_PATH)

X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")  

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model

MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)

print(f"Model saved to {MODEL_PATH}")