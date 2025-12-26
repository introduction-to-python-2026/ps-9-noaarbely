import pandas as pd
df = pd.read_csv("penguins.csv")
# df.dropna()
df.head()

import seaborn as sns 
import matplotlib.pyplot as plt

# Drop  rows with missing 'sex' values fot the pairplot, as it's the targer variable
df_cleaned = df.dropna(subset=['sex'])

x = df_cleaned[['bill_depthmm' , 'body_mass_g']]
y=df_cleaned[['sex']]

#x.head()
#y.head()

sns.pairplot(df_cleaned, vars=['bill_depth_mm', 'body_mass_g'], hue='sex')
plt.suptitle('Pair Plot of Numerical Features by Sex', y=1.02)
plt.show()

from sklearn.preprocessing import MinMaxScal

scaler = MinMaxScaler()
x_scale = scaler.fit_transform(x) #normalize the data
len(x_scale)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train , y_test = train_test_split(x_scale, y, test_size=0.2)
len(x_train), len(x_test)

#from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#dt = DecisionTreeClassifier(max_depth=2)
dt = KNeighborsClassifier(n_neighbors=3)
dt.fit(x_train, y_train)

from sklearn.metrics import accuracy_score

y_predict = dt.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
accuracy

import joblib

joblib.dump(dt, 'knn_penguins.joblib')
model = joblib.load('knn_penguins.joblib')
