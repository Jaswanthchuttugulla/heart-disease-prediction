import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier

# ==============================
# 1. Load Combined Dataset
# ==============================
DATA_PATH = "dataset/heart-disease-dataset.csv"
df = pd.read_csv(DATA_PATH)

print("Combined dataset loaded:")
print(df.shape)
print(df.head())

# ==============================
# 2. Split Features & Target
# ==============================
X = df.drop("target", axis=1)   # use "target" if merged file uses that column
y = df["target"]

# ==============================
# 3. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# 4. Build & Train Model
# ==============================
model = XGBClassifier(
    n_estimators=1000,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==============================
# 5. Evaluate
# ==============================
y_probs = model.predict_proba(X_test)[:,1]
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_probs))

# ==============================
# 6. Save Model
# ==============================
with open("model_combined.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved as model_combined.pkl")
