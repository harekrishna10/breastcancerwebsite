import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🎗️",
    layout="centered"
)


# --------------------------------------------------
# LOAD MODEL AND SCALER
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("svm_model.pkl")
    scaler = joblib.load("scaler.pkl")

    return model, scaler

model, scaler = load_model()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

if os.path.exists("cancer_image.png"):
    st.image("cancer_image.png", width=500)

st.title("🎗️ Breast Cancer Prediction")

st.write(
    """
    Enter the patient's tumor measurements below.
    The application uses a **Support Vector Machine (SVM)**
    model to predict whether the tumor is likely to be
    **Benign or Malignant**.
    """
)

st.divider()


# --------------------------------------------------
# INPUT FEATURES
# --------------------------------------------------

st.subheader("Enter Tumor Measurements")


# Mean features
st.markdown("### 📊 Mean Features")

col1, col2 = st.columns(2)

with col1:
    radius_mean = st.number_input(
        "Radius Mean",
        min_value=0.0,
        value=14.0,
        step=0.1
    )

    perimeter_mean = st.number_input(
        "Perimeter Mean",
        min_value=0.0,
        value=90.0,
        step=0.1
    )

    smoothness_mean = st.number_input(
        "Smoothness Mean",
        min_value=0.0,
        value=0.10,
        step=0.01
    )

    concavity_mean = st.number_input(
        "Concavity Mean",
        min_value=0.0,
        value=0.08,
        step=0.01
    )

    symmetry_mean = st.number_input(
        "Symmetry Mean",
        min_value=0.0,
        value=0.18,
        step=0.01
    )

with col2:
    texture_mean = st.number_input(
        "Texture Mean",
        min_value=0.0,
        value=19.0,
        step=0.1
    )

    area_mean = st.number_input(
        "Area Mean",
        min_value=0.0,
        value=650.0,
        step=1.0
    )

    compactness_mean = st.number_input(
        "Compactness Mean",
        min_value=0.0,
        value=0.10,
        step=0.01
    )

    concave_points_mean = st.number_input(
        "Concave Points Mean",
        min_value=0.0,
        value=0.05,
        step=0.01
    )

    fractal_dimension_mean = st.number_input(
        "Fractal Dimension Mean",
        min_value=0.0,
        value=0.06,
        step=0.001
    )


# --------------------------------------------------
# SE FEATURES
# --------------------------------------------------

st.markdown("### 📈 Standard Error Features")

col1, col2 = st.columns(2)

with col1:
    radius_se = st.number_input(
        "Radius SE",
        min_value=0.0,
        value=0.4,
        step=0.01
    )

    perimeter_se = st.number_input(
        "Perimeter SE",
        min_value=0.0,
        value=2.8,
        step=0.1
    )

    smoothness_se = st.number_input(
        "Smoothness SE",
        min_value=0.0,
        value=0.007,
        step=0.001
    )

    concavity_se = st.number_input(
        "Concavity SE",
        min_value=0.0,
        value=0.02,
        step=0.01
    )

    symmetry_se = st.number_input(
        "Symmetry SE",
        min_value=0.0,
        value=0.02,
        step=0.01
    )

with col2:
    texture_se = st.number_input(
        "Texture SE",
        min_value=0.0,
        value=1.2,
        step=0.1
    )

    area_se = st.number_input(
        "Area SE",
        min_value=0.0,
        value=40.0,
        step=1.0
    )

    compactness_se = st.number_input(
        "Compactness SE",
        min_value=0.0,
        value=0.02,
        step=0.01
    )

    concave_points_se = st.number_input(
        "Concave Points SE",
        min_value=0.0,
        value=0.01,
        step=0.01
    )

    fractal_dimension_se = st.number_input(
        "Fractal Dimension SE",
        min_value=0.0,
        value=0.003,
        step=0.001
    )


# --------------------------------------------------
# WORST FEATURES
# --------------------------------------------------

st.markdown("### 🔬 Worst Features")

col1, col2 = st.columns(2)

with col1:
    radius_worst = st.number_input(
        "Radius Worst",
        min_value=0.0,
        value=16.0,
        step=0.1
    )

    perimeter_worst = st.number_input(
        "Perimeter Worst",
        min_value=0.0,
        value=105.0,
        step=0.1
    )

    smoothness_worst = st.number_input(
        "Smoothness Worst",
        min_value=0.0,
        value=0.13,
        step=0.01
    )

    concavity_worst = st.number_input(
        "Concavity Worst",
        min_value=0.0,
        value=0.2,
        step=0.01
    )

    symmetry_worst = st.number_input(
        "Symmetry Worst",
        min_value=0.0,
        value=0.29,
        step=0.01
    )

with col2:
    texture_worst = st.number_input(
        "Texture Worst",
        min_value=0.0,
        value=25.0,
        step=0.1
    )

    area_worst = st.number_input(
        "Area Worst",
        min_value=0.0,
        value=850.0,
        step=1.0
    )

    compactness_worst = st.number_input(
        "Compactness Worst",
        min_value=0.0,
        value=0.25,
        step=0.01
    )

    concave_points_worst = st.number_input(
        "Concave Points Worst",
        min_value=0.0,
        value=0.10,
        step=0.01
    )

    fractal_dimension_worst = st.number_input(
        "Fractal Dimension Worst",
        min_value=0.0,
        value=0.08,
        step=0.001
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

st.divider()

if st.button("🔍 Predict", use_container_width=True):

    input_data = np.array([[
        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean,
        smoothness_mean,
        compactness_mean,
        concavity_mean,
        concave_points_mean,
        symmetry_mean,
        fractal_dimension_mean,

        radius_se,
        texture_se,
        perimeter_se,
        area_se,
        smoothness_se,
        compactness_se,
        concavity_se,
        concave_points_se,
        symmetry_se,
        fractal_dimension_se,

        radius_worst,
        texture_worst,
        perimeter_worst,
        area_worst,
        smoothness_worst,
        compactness_worst,
        concavity_worst,
        concave_points_worst,
        symmetry_worst,
        fractal_dimension_worst
    ]])

    # Convert input into DataFrame
    input_df = pd.DataFrame(
        input_data,
        columns=[
            "radius_mean",
            "texture_mean",
            "perimeter_mean",
            "area_mean",
            "smoothness_mean",
            "compactness_mean",
            "concavity_mean",
            "concave points_mean",
            "symmetry_mean",
            "fractal_dimension_mean",

            "radius_se",
            "texture_se",
            "perimeter_se",
            "area_se",
            "smoothness_se",
            "compactness_se",
            "concavity_se",
            "concave points_se",
            "symmetry_se",
            "fractal_dimension_se",

            "radius_worst",
            "texture_worst",
            "perimeter_worst",
            "area_worst",
            "smoothness_worst",
            "compactness_worst",
            "concavity_worst",
            "concave points_worst",
            "symmetry_worst",
            "fractal_dimension_worst"
        ]
    )

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Probability
    probability = model.predict_proba(input_scaled)[0]

    confidence = max(probability) * 100


    # --------------------------------------------------
    # RESULT
    # --------------------------------------------------

    if prediction == 1:

        st.error(
            f"⚠️ Malignant tumor predicted "
            f"(confidence: {confidence:.1f}%)"
        )

        st.progress(int(confidence))

    else:

        st.success(
            f"✅ Benign tumor predicted "
            f"(confidence: {confidence:.1f}%)"
        )

        st.progress(int(confidence))


    st.write(f"**Prediction confidence:** {confidence:.1f}%")

    st.warning(
        """
        This prediction is for educational purposes only.
        It is not a medical diagnosis. Please consult a
        qualified healthcare professional for medical advice.
        """
    )