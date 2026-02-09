from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import os

os.makedirs("models", exist_ok=True)

# Dummy training data
X = np.random.rand(200, 4)
y = np.random.randint(0, 2, 200)

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

logistic = LogisticRegression(max_iter=1000).fit(X_scaled, y)
rf = RandomForestClassifier(n_estimators=50).fit(X, y)
svm = SVC(kernel="linear", probability=True).fit(X_scaled, y)
iso = IsolationForest(contamination=0.05).fit(X_scaled)

joblib.dump(logistic, "models/logistic.pkl")
joblib.dump(rf, "models/random_forest.pkl")
joblib.dump(svm, "models/svm.pkl")
joblib.dump(iso, "models/isolation_forest.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Dummy model files created inside models/")

