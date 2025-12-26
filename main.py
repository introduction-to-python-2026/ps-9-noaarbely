import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1) Load data
df = pd.read_csv("penguins.csv")

# 2) Clean missing values in target
df_cleaned = df.dropna(subset=["sex"])

# 3) Select features + target (שימי לב לשמות העמודות!)
X = df_cleaned[["bill_depth_mm", "body_mass_g"]]
y = df_cleaned["sex"] # עדיף כ-Series ולא כ-DataFrame

# (Optional) EDA
sns.pairplot(df_cleaned, vars=["bill_depth_mm", "body_mass_g"], hue="sex")
plt.suptitle("Pair Plot of Numerical Features by Sex", y=1.02)
plt.show()

# 4) Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 5) Split data (מומלץ stratify כדי לשמור יחס מחלקות)
X_train, X_test, y_train, y_test = train_test_split(
X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 6) Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 7) Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 8) Save model
joblib.dump(model, "knn_penguins.joblib")
loaded_model = joblib.load("knn_penguins.joblib")
