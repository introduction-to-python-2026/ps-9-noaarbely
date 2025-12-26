import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("penguins.csv")

df_cleaned = df.dropna(subset=["sex"])
X = df_cleaned[["bill_depth_mm", "body_mass_g"]]
y = df_cleaned["sex"]

sns.pairplot(df_cleaned, vars=["bill_depth_mm", "body_mass_g"], hue="sex")
plt.suptitle("Pair Plot of Numerical Features by Sex", y=1.02)
plt.show()

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

joblib.dump(model, "knn_penguins.joblib")

config_content = """selected_features: ["bill_depth_mm", "body_mass_g"]
path: "knn_penguins.joblib"
"""
with open("config.yaml", "w", encoding="utf-8") as f:
f.write(config_content)

loaded_model = joblib.load("knn_penguins.joblib")
