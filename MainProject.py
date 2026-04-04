import pandas as pd
import numpy as np
import joblib
import os

from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =========================================================
# CONFIGURATION
# =========================================================
CSV_FILE = "Student_marks.csv"
MODEL_FILE = "student_exam_model.pkl"
TARGET_COLUMN = "exam_score"

# =========================================================
# TRAIN MODEL FUNCTION
# =========================================================
def train_and_save_model():
    print("📂 Loading dataset...")
    df = pd.read_csv(CSV_FILE)

    print("✅ Dataset Loaded Successfully!")
    print("Shape:", df.shape)

    if "student_id" in df.columns:
        df.drop(columns=["student_id"], inplace=True)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numerical_features = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n🚀 Training model...")
    pipeline.fit(X_train, y_train)
    print("✅ Model Training Completed!")

    y_pred = pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n================ MODEL PERFORMANCE ================")
    print(f"R2 Score   : {r2:.4f}")
    print(f"MAE        : {mae:.4f}")
    print(f"RMSE       : {rmse:.4f}")
    print("==================================================")

    joblib.dump(pipeline, MODEL_FILE)
    print(f"\n💾 Model saved as '{MODEL_FILE}'")

    return pipeline

# =========================================================
# LOAD OR TRAIN MODEL
# =========================================================
if os.path.exists(MODEL_FILE):
    print(f"📦 Loading existing model from '{MODEL_FILE}'...")
    model = joblib.load(MODEL_FILE)
    print("✅ Model Loaded Successfully!")
else:
    model = train_and_save_model()

# =========================================================
# FLASK APP
# =========================================================
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_form", methods=["POST"])
def predict_form():
    try:
        input_data = pd.DataFrame([{
            "age": int(request.form["age"]),
            "gender": request.form["gender"],
            "academic_level": request.form["academic_level"],
            "study_hours": float(request.form["study_hours"]),
            "self_study_hours": float(request.form["self_study_hours"]),
            "online_classes_hours": float(request.form["online_classes_hours"]),
            "social_media_hours": float(request.form["social_media_hours"]),
            "gaming_hours": float(request.form["gaming_hours"]),
            "sleep_hours": float(request.form["sleep_hours"]),
            "screen_time_hours": float(request.form["screen_time_hours"]),
            "exercise_minutes": int(request.form["exercise_minutes"]),
            "caffeine_intake_mg": int(request.form["caffeine_intake_mg"]),
            "part_time_job": int(request.form["part_time_job"]),
            "upcoming_deadline": int(request.form["upcoming_deadline"]),
            "internet_quality": request.form["internet_quality"],
            "mental_health_score": float(request.form["mental_health_score"]),
            "focus_index": float(request.form["focus_index"]),
            "burnout_level": float(request.form["burnout_level"]),
            "productivity_score": float(request.form["productivity_score"])
        }])

        prediction = model.predict(input_data)[0]

        return render_template("index.html", prediction=round(float(prediction), 2))

    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        input_data = pd.DataFrame([data])

        prediction = model.predict(input_data)[0]

        return jsonify({
            "predicted_exam_score": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)