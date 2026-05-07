import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# ==============================
# 1. Load Dataset
# ==============================
DATA_PATH = "dataset/heart-disease-dataset.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded Successfully")
print(df.shape)
print(df.head())

# ==============================
# 2. Features and Target
# ==============================
X = df.drop("target", axis=1)
y = df["target"]

# ==============================
# 3. Train Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# 4. Build Model
# ==============================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

# ==============================
# 5. Train Model
# ==============================
model.fit(X_train, y_train)

# ==============================
# 6. Predictions
# ==============================
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

# ==============================
# 7. Evaluation
# ==============================
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nROC-AUC Score:", roc_auc_score(y_test, y_probs))

# ==============================
# 8. Save Model
# ==============================
joblib.dump(model, "model.pkl")

print("\nModel saved successfully as model.pkl")