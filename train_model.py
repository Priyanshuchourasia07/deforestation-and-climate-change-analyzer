import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_excel("C:/Deforestation_Project/dataset/Deforestation_Climate_Dataset_900x15.xlsx")

# Drop non-numeric column (Country)
df = df.drop(columns=["Country"])

# Features (X) and Target (y)
X = df.drop(columns=["Temperature"])
y = df["Temperature"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling (important for ML models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Example prediction
sample = X_test[0].reshape(1, -1)
print("Predicted Temperature:", model.predict(sample))