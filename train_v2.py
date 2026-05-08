import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("balanced_india_dataset.csv")

df['acq_date'] = pd.to_datetime(df['acq_date'])
df['month'] = df['acq_date'].dt.month

df['Month_Sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['Month_Cos'] = np.cos(2 * np.pi * df['month'] / 12)

features = ['temp_c', 'humidity', 'wind_kmh']
target = 'fire_occurred'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(" Training the new Brain (XGBoost Classifier)...")
model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f" Model Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

joblib.dump(model, "models/forest_fire_v2.pkl")

