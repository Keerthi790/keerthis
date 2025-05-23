# multi_disease_model_app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# === Global ===
DATASETS = {
    "diabetes": ["diabetes.csv"],
    "heart":    ["heart.csv"],
    "kidney":   ["kidney_disease.csv", "kidney.csv"],
    "liver":    ["Indian Liver Patient Dataset (ILPD).csv", "liver.csv"]
}

models = {}
scalers = {}

# === Helper Functions ===

def locate_file(candidates):
    for fn in candidates:
        if os.path.exists(fn):
            return fn
    raise FileNotFoundError(f"❌ Dataset file not found. Expected one of: {candidates}")

def preprocess_data(name, df):
    df = df.copy()
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)

    if name == "kidney":
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.lower()
        if 'classification' in df.columns:
            df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})
        df = df.select_dtypes(include=[np.number])

    elif name == "liver":
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        if 'Dataset' in df.columns:
            df['Dataset'] = df['Dataset'].replace({1: 1, 2: 0})

    return df

def get_target_column(name, df):
    candidates = {
        "diabetes": [df.columns[-1]],
        "heart":    [df.columns[-1]],
        "kidney":   ["classification", "class", "target"],
        "liver":    ["Dataset", "class", "target"]
    }
    for col in candidates[name]:
        if col in df.columns:
            return col
    return df.columns[-1]

def train_one(name):
    path = locate_file(DATASETS[name])
    df = pd.read_csv(path)
    df = preprocess_data(name, df)

    target = get_target_column(name, df)
    X = df.drop(columns=[target])
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ {name.capitalize()} model trained with accuracy: {acc:.2f}")

    models[name] = model
    scalers[name] = scaler

def load_and_train_all():
    for disease in DATASETS:
        train_one(disease)

def predict_disease(name, inputs):
    if name not in models:
        raise ValueError(f"Model for '{name}' not loaded.")
    model = models[name]
    scaler = scalers[name]
    return model.predict(scaler.transform([inputs]))[0]

def get_feature_info(name):
    path = locate_file(DATASETS[name])
    df = pd.read_csv(path)
    df = preprocess_data(name, df)
    target = get_target_column(name, df)
    X = df.drop(columns=[target])
    return list(X.columns)

# === Streamlit App ===

st.set_page_config(page_title="Multi-Disease Prediction", layout="centered")
st.title("🧠 Multi-Disease Prediction System")

# Load models once
if 'models_loaded' not in st.session_state:
    load_and_train_all()
    st.session_state.models_loaded = True

# Sidebar for disease selection
selected_disease = st.sidebar.radio(
    "Select Disease",
    options=["diabetes", "heart", "kidney", "liver"],
    format_func=lambda x: x.capitalize() + " Disease Prediction"
)

st.subheader(f"Enter values for {selected_disease.capitalize()} Disease Prediction")

# Dynamic input fields
features = get_feature_info(selected_disease)
user_inputs = []

for feature in features:
    val = st.number_input(label=feature, value=0.0, step=0.1)
    user_inputs.append(val)

# Prediction button
if st.button("Predict"):
    result = predict_disease(selected_disease, user_inputs)
    if result == 1:
        st.error(f"⚠️ Positive for {selected_disease.capitalize()} Disease")
    else:
        st.success(f"✅ Negative for {selected_disease.capitalize()} Disease")
