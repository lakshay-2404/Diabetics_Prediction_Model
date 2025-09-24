import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Paths
MODEL_PATH = "diabetes_model.pkl"
DATA_PATH = "diabetes.csv"   # original dataset
USER_DATA_PATH = "user_data.csv"

# Load base dataset
df = pd.read_csv(DATA_PATH)

# Load model if exists, else train new
try:
    model = joblib.load(MODEL_PATH)
except:
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

st.title("🩺 Diabetes Prediction & Continuous Learning")

st.write("Enter details to predict and also update the training data:")

# Collect user input
pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose Level", 0, 300, 120)
blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, format="%.2f")
age = st.number_input("Age", 0, 120, 30)

# true label (if known, for retraining)
outcome = st.selectbox("Do you know the true outcome?", ["Unknown", "Non-Diabetic (0)", "Diabetic (1)"])

input_data = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [dpf],
    'Age': [age]
})

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ Predicted: **Diabetic**")
    else:
        st.success("✅ Predicted: **Non-Diabetic**")

# Retraining option
if outcome != "Unknown" and st.button("Update Model with This Entry"):
    # Save input with true label
    input_data["Outcome"] = 0 if "Non-Diabetic" in outcome else 1
    input_data.to_csv(USER_DATA_PATH, mode="a", header=not pd.io.common.file_exists(USER_DATA_PATH), index=False)

    # Reload all data (original + user)
    user_df = pd.read_csv(USER_DATA_PATH) if pd.io.common.file_exists(USER_DATA_PATH) else pd.DataFrame()
    combined_df = pd.concat([df, user_df], ignore_index=True)

    # Retrain
    X = combined_df.drop("Outcome", axis=1)
    y = combined_df["Outcome"]
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    # Save updated model
    joblib.dump(model, MODEL_PATH)

    st.success("✅ Model retrained and saved with new data!")
