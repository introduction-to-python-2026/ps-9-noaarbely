import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("penguins.csv")

df = df.dropna(subset=["sex"])

X = df[["bill_depth_mm", "body_mass_g"]]
y = df["sex"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

joblib.dump(model, "knn_penguins.joblib")

with open("config.yaml", "w") as f:
    f.write(
        'selected_features: ["bill_depth_mm", "body_mass_g"]\n'
        'path: "knn_penguins.joblib"\n'
    )
