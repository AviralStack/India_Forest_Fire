import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. LOAD THE NEW DATASET
df = pd.read_csv("balanced_india_dataset.csv")
print(f" Loaded {len(df)} rows of Real-World Data.")

# 2. FEATURE ENGINEERING (The 'Cyclical Month' Trick)
# We need to turn "2020-10-26" into Numbers the AI understands
df['acq_date'] = pd.to_datetime(df['acq_date'])
df['month'] = df['acq_date'].dt.month

# Convert Month to Cycle (So Dec and Jan are close to each other)
df['Month_Sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['Month_Cos'] = np.cos(2 * np.pi * df['month'] / 12)

# 3. DEFINE FEATURES & TARGET
# Notice: We are using LAT/LON now! The model will learn geography.
features = ['temp_c', 'humidity', 'wind_kmh']
target = 'fire_occurred'

X = df[features]
y = df[target]

# 4. SPLIT DATA (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. TRAIN THE CLASSIFIER
print(" Training the new Brain (XGBoost Classifier)...")
model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# 6. EVALUATE
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f" Model Accuracy: {accuracy * 100:.2f}%")
print("Detailed Report:")
print(classification_report(y_test, y_pred))

# 7. SAVE THE MODEL
# We save it with a new name so we don't overwrite the old one just yet
joblib.dump(model, "models/forest_fire_v2.pkl")
print("Saved to models/forest_fire_v2.pkl")