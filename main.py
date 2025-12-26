import pandas as pd

df = pd.read_csv("penguins.csv")
df.head()

import seaborn as sns
import matplotlib.pyplot as plt

df_cleaned = df.dropna(subset=['sex'])
x = df_cleaned[['bill_depth_mm', 'body_mass_g']]
y = df_cleaned['sex']

sns.pairplot(df_cleaned, vars=['bill_depth_mm', 'body_mass_g'], hue='sex')
plt.suptitle('Pair Plot of Numerical Features by Sex', y=1.02)
plt.show()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x_scale = scaler.fit_transform(x)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier

dt = KNeighborsClassifier(n_neighbors=3)
dt.fit(x_train, y_train)

from sklearn.metrics import accuracy_score

y_predict = dt.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
print(f"Model Accuracy: {accuracy}")

import joblib

joblib.dump(dt, 'knn_penguins.joblib')

config_content = """
selected_features: ["bill_depth_mm", "body_mass_g"]  
path: "knn_penguins.joblib"  
"""
with open('config.yaml', 'w') as f:
    f.write(config_content)

model = joblib.load('knn_penguins.joblib')
