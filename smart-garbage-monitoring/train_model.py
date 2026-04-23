import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

df = pd.read_csv("data/garbage_large_dataset.csv")

X = df[["Fill_Level", "Temperature", "Humidity"]]
y = df["Status"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Advanced Algorithm
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42
)

model.fit(X_train, y_train)

# Anomaly Detection
anomaly_model = IsolationForest(contamination=0.02)
anomaly_model.fit(X)

print(classification_report(y_test, model.predict(X_test)))

joblib.dump((model, anomaly_model, encoder), "models/garbage_model.pkl")

print("Model trained and saved successfully!")
