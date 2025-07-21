import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(page_title="AI Diabetes Risk Assessment", layout="centered")
st.title("🧠 AI Diabetes Risk Assessment")

MODEL_PATH = "Diabetesmodel.pkl"
FEATURES_PATH = "Diabetesmodel_features.pkl"

# Sidebar mode selection
mode = st.sidebar.radio("Choose Mode", ["Use Prediction", "Model Info"])

# ------------------------ PREDICT MODE ------------------------
if mode == "Use Prediction":
    st.header("📈 Predict Diabetes Risk")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        st.error("❌ Pretrained model or its feature list not found. Please ensure both `Diabetesmodel.pkl` and `Diabetesmodel_features.pkl` exist.")
        st.stop()

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        feature_names = pickle.load(f)

        st.subheader("Enter Medical Data")

    user_input = []

    sliders_config = {
        "Pregnancy": {"min": 0, "max": 12, "default": 2, "step": 1, "format": "%d"},
        "Glucose": {"min": 50, "max": 200, "default": 120, "step": 1, "format": "%d"},
        "BloodPressure": {"min": 30, "max": 140, "default": 70, "step": 1, "format": "%d"},
        "SkinThickness": {"min": 0, "max": 100, "default": 20, "step": 1, "format": "%d"},
        "Insulin": {"min": 0, "max": 800, "default": 85, "step": 1, "format": "%d"},
        "BMI": {"min": 10.0, "max": 60.0, "default": 28.0, "step": 0.1, "format": "%.1f"},
        "DiabetesPedigreeFunction": {"min": 0.0, "max": 2.5, "default": 0.5, "step": 0.001, "format": "%.3f"},
        "Age": {"min": 10, "max": 100, "default": 33, "step": 1, "format": "%d"}
    }

    units = {
        "Pregnancy": "(Months)",
        "Glucose": "(mg/dL)",
        "BloodPressure": "(mm Hg)",
        "SkinThickness": "(mm)",
        "Insulin": "(mu U/ml)",
        "BMI": "(kg/m²)",
        "DiabetesPedigreeFunction": "(score)",
        "Age": "(years)"
    }

         # Feature name mapping to support display
    rename_map = {
        "Pregnancies": "Pregnancy"
    }

    for original_feature in feature_names:
        display_feature = rename_map.get(original_feature, original_feature)
        config = sliders_config.get(display_feature, {})
        label = f"{display_feature} {units.get(display_feature, '')}"

        # Set correct type for values
        if isinstance(config["min"], float) or isinstance(config["max"], float):
            step = float(config["step"])
            min_val = float(config["min"])
            max_val = float(config["max"])
            default_val = float(config["default"])
        else:
            step = int(config["step"])
            min_val = int(config["min"])
            max_val = int(config["max"])
            default_val = int(config["default"])

        # Manual input only
        val = st.number_input(
            label,
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=step,
            format=config["format"]
        )
        user_input.append(val)

    if st.button("Predict"):
        try:
            input_array = np.array(user_input).reshape(1, -1)
            result = model.predict(input_array)
            if result[0] == 1:
                st.error("🩺 You are likely to have diabetes.")
            else:
                st.success("✅ You are safe.")
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")

# ------------------------ MODEL INFO MODE ------------------------
elif mode == "Model Info":

    st.markdown("""
    ### About the Model
    This model is trained using the **Pima Indians Diabetes Dataset**.

    **Model Performance (on test set):**
    - Accuracy: **~85%**
    - F1 Score: **~0.77**
    - AUC Score: **~0.89**

    > By: Wong Jia Hern
    """)

    st.info("⚙️ Model file used: `Diabetesmodel.pkl`")
