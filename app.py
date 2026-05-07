import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Title
st.title("🌍 Deforestation Climate Predictor")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_excel("C:/Deforestation_Project/dataset/Deforestation_Climate_Dataset_900x15.xlsx")
    df = df.drop(columns=["Country"])
    return df

df = load_data()

st.write("### Dataset Preview")
st.dataframe(df.head())

# Features & Target
X = df.drop(columns=["Temperature"])
y = df["Temperature"]

# Train model
@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler

model, scaler = train_model()

st.write("## 🔮 Predict Temperature")

# User input sliders
user_input = []
for col in X.columns:
    val = st.slider(f"{col}", float(X[col].min()), float(X[col].max()))
    user_input.append(val)

# Prediction button
if st.button("Predict"):
    input_df = pd.DataFrame([user_input], columns=X.columns)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    st.success(f"🌡️ Predicted Temperature: {prediction[0]:.2f}")