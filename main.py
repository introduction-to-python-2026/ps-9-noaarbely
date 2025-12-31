from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import lab_setup_do_not_edit
import pandas as pd
import joblib


# Loads the CSV file and creates a DataFrame object
df = pd.read_csv('/content/parkinsons.csv')

# Chooses features to analyze
input_features = ['MDVP:APQ', 'MDVP:Shimmer(dB)']
output_feature = 'status'

# Extracts the chosen features out of the DataFrame
X = df[input_features]
y = df[output_feature]

# Min-Max scaling of the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Divides the dataset into a training set and a validation set
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Creates a KNN model object and trains it
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Evaluates the model's accuracy on the test set.
# Checks if the accuracy is at least 0.8
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
if accuracy >= 0.8:
    print("Accuracy target of 0.8 met or exceeded!")
else:
    print("Accuracy is below 0.8. Consider adjusting the model or features.")

# Saves the model as a .joblib file
joblib.dump(model, 'parkinson_model.joblib')
