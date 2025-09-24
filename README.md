

# Diabetes Prediction & Continuous Learning (Streamlit)

A Streamlit app that predicts diabetes from clinical inputs using a RandomForest model and supports continuous learning by appending labeled user entries, retraining, and persisting the updated model.

## Features
- Real‑time prediction from 8 inputs with class label and risk probability using a trained RandomForest.
- Continuous learning workflow: append labeled user entries, retrain on combined data, and save an updated model file.
- Script‑relative paths for reliable file I/O across environments.

## Project structure
- app.py — Streamlit app with prediction and retraining logic; loads/saves model via joblib and manages user_data.csv.
- diabetes.csv — Base dataset with columns [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome]. Can be found on Kaggle.
- diabetes_model.pkl — Persisted RandomForest model created on first run (auto‑generated).
- user_data.csv — Appended labeled rows from interactive usage (auto‑created).
- Diabetes_prediction.ipynb — Supporting notebook for exploration/feature engineering (optional).

## Getting started

### Prerequisites
- Python 3.10+ and pip.
- Recommended packages: streamlit, pandas, scikit‑learn, joblib, numpy.

Example requirements.txt:
```
streamlit
pandas
scikit-learn
joblib
numpy
```

### Setup
1) Place app.py and diabetes.csv in the same folder. Paths in the app are script‑relative by default.  
2) Install dependencies:
```
pip install -r requirements.txt
```
3) Run the app:
```
streamlit run app.py
```

On first launch, the app will train a RandomForest on diabetes.csv and save diabetes_model.pkl; subsequent runs will load the saved model.

## Usage
- Fill inputs: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age. Values are clipped to reasonable ranges to reduce out‑of‑distribution effects.
- Click Predict to see:
  - Predicted class: Diabetic or Non‑Diabetic.
  - Risk probability if available.
- If the true outcome is known:
  - Select “Non‑Diabetic (0)” or “Diabetic (1)” and click “Update Model with This Entry”. The row is appended to user_data.csv, the model retrains on base+user data, and diabetes_model.pkl is updated.

## Data schema
- Pregnancies (int, 0–20)
- Glucose (int, 0–300)
- BloodPressure (int, 0–200)
- SkinThickness (int, 0–100)
- Insulin (int, 0–900)
- BMI (float, 0–70)
- DiabetesPedigreeFunction (float, 0–3)
- Age (int, 0–120)
- Outcome (int, {0,1}) in base/user data for training

## Model
- Estimator: RandomForestClassifier(random_state=42).
- Training on first run: X = df.drop("Outcome"), y = df["Outcome"] from diabetes.csv.
- Persistence: joblib.dump(model, "diabetes_model.pkl"); joblib.load on subsequent runs.
- Retraining: on base + appended user_data.csv whenever a labeled entry is submitted.

## Notes and recommendations
- Probability display uses predict_proba when available to communicate risk confidence.
- Inputs are clipped to pre‑defined ranges; consider adding dataset‑driven validation or imputation if extending preprocessing.
- If enhancing features (e.g., scaling or imputers), wrap preprocessing + model in a single Pipeline and persist the pipeline instead of a bare estimator.

## Troubleshooting
- FileNotFoundError for base CSV: ensure diabetes.csv resides next to app.py; the app expects script‑relative paths.
- No retraining after prediction: retraining only triggers when a true outcome is selected (not “Unknown”) and “Update Model with This Entry” is clicked.
- Schema mismatch: user_data.csv must include the same columns as the base dataset including Outcome; the app writes the correct schema automatically on first labeled update.

## License
Add your preferred license (e.g., MIT) and ensure dataset usage complies with its original terms.

## Acknowledgements
This app uses the Pima Indians Diabetes dataset schema (as provided in diabetes.csv) and scikit‑learn’s RandomForestClassifier. Credit the dataset’s source if publishing publicly.

[1](https://discuss.streamlit.io/t/streamlit-best-practices/57921)
[2](https://blog.streamlit.io/best-practices-for-building-genai-apps-with-streamlit/)
[3](https://docs.streamlit.io/develop/concepts/connections/connecting-to-data)
[4](https://docs.streamlit.io)
[5](https://docs.healthuniverse.com/overview/building-apps-in-health-universe/developing-your-health-universe-app/working-in-streamlit/streamlit-best-practices)
[6](https://deepnote.com/blog/ultimate-guide-to-the-streamlit-library)
[7](https://docs.snowflake.com/en/developer-guide/streamlit/getting-started)
[8](https://docs.streamlit.io/develop/concepts/multipage-apps/overview)
