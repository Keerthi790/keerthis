# app.py

import streamlit as st
import multi_disease_model as mdm

st.set_page_config(page_title="Multi Disease Prediction", layout="centered")
st.title("🩺 Multiple Disease Prediction System")

# Sidebar
st.sidebar.title("Select Disease")
choice = st.sidebar.radio("", [
    "Diabetes Prediction",
    "Heart Disease Prediction",
    "Kidney Disease Prediction",
    "Liver Disease Prediction"
])

# Load models only once
if 'loaded' not in st.session_state:
    try:
        mdm.load_and_train_all()
        st.session_state.loaded = True
    except FileNotFoundError as e:
        st.sidebar.error(str(e))
        st.stop()


# Render form without showing ranges
def render_form(key_name, disease_key):
    st.header(f"{disease_key.capitalize()} Prediction")
    features = mdm.get_feature_info(disease_key)
    inputs = []

    with st.form(key_name):
        for feat in features:
            val = st.number_input(label=feat, value=0.0)
            inputs.append(val)
        submit = st.form_submit_button("Predict")# app.py

import streamlit as st
import multi_disease_model as mdm

st.set_page_config(page_title="Multi Disease Prediction", layout="centered")
st.title("🩺 Multiple Disease Prediction System")

# Sidebar
st.sidebar.title("Select Disease")
choice = st.sidebar.radio("", [
    "Diabetes Prediction",
    "Heart Disease Prediction",
    "Kidney Disease Prediction",
    "Liver Disease Prediction"
])

# Load models only once
if 'loaded' not in st.session_state:
    try:
        mdm.load_and_train_all()
        st.session_state.loaded = True
    except FileNotFoundError as e:
        st.sidebar.error(str(e))
        st.stop()

# Render form without showing ranges
def render_form(key_name, disease_key):
    st.header(f"{disease_key.capitalize()} Prediction")
    features = mdm.get_feature_info(disease_key)
    inputs = []

    with st.form(key_name):
        for feat in features:
            val = st.number_input(label=feat, value=0.0)
            inputs.append(val)
        submit = st.form_submit_button("Predict")
    
    if submit:
        prediction = mdm.predict_disease(disease_key, inputs)
        if prediction == 1:
            st.error(f"⚠️ Prediction: Positive for {disease_key.capitalize()} Disease")
        else:
            st.success(f"✅ Prediction: Negative for {disease_key.capitalize()} Disease")

# Routing based on sidebar selection
if choice == "Diabetes Prediction":
    render_form("form_diabetes", "diabetes")
elif choice == "Heart Disease Prediction":
    render_form("form_heart", "heart")
elif choice == "Kidney Disease Prediction":
    render_form("form_kidney", "kidney")
elif choice == "Liver Disease Prediction":
    render_form("form_liver", "liver")


    if submit:
        prediction = mdm.predict_disease(disease_key, inputs)
        if prediction == 1:
            st.error(f"⚠️ Prediction: Positive for {disease_key.capitalize()} Disease")
        else:
            st.success(f"✅ Prediction: Negative for {disease_key.capitalize()} Disease")


# Routing based on sidebar selection
if choice == "Diabetes Prediction":
    render_form("form_diabetes", "diabetes")
elif choice == "Heart Disease Prediction":
    render_form("form_heart", "heart")
elif choice == "Kidney Disease Prediction":
    render_form("form_kidney", "kidney")
elif choice == "Liver Disease Prediction":
    render_form("form_liver", "liver")
