import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json

# Load dataset (FIXED)
data = pd.read_csv(
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/winequality-red.csv",
    header=None
)

data.columns = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "quality"
]

print("Columns:", data.columns)

X = data.drop("quality", axis=1)
y = data["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("MSE:", mse)
print("R2:", r2)

joblib.dump(model, "model.pkl")

with open("metrics.json", "w") as f:
    json.dump({"mse": mse, "r2": r2}, f)