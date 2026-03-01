# STEP 1: Import Libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# STEP 2: Load Dataset
df = pd.read_csv(r"D:\Project\student_marks (1).csv")
print("Dataset Preview:")
print(df.head())

# STEP 3: Convert Pass/Fail → 1/0
df["Result"] = df["Result"].map({"Fail": 0, "Pass": 1})

# STEP 4: Select Features & Target
X = df[["StudyHours", "Attendance (%)"]]
y = df["Result"]

# STEP 5: Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 6: Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# STEP 7: Prediction

y_pred = model.predict(X_test)

# STEP 8: Accuracy Check

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

# STEP 9: Predict New Student

study_hours = 5
attendance = 80

prediction = model.predict([[study_hours, attendance]])

if prediction[0] == 1:
    print("\nStudent Result: PASS")
else:
    print("\nStudent Result: FAIL")