## 🎗️ Breast Cancer Prediction Website

A Machine Learning web application that predicts whether a breast tumor is **Benign** or **Malignant** using a **Support Vector Machine (SVM)** classification model.

The application is developed using Python and Streamlit and allows users to enter tumor measurements and receive a prediction from the trained SVM model.

## 🌐 Live Demo

https://breastcancerwebsites.streamlit.app/

## 🧠 Machine Learning Model

This project uses a **Support Vector Machine (SVM)** with a linear kernel for binary classification.

The model predicts:

- 🟢 Benign
- 🔴 Malignant

Feature scaling is performed using `StandardScaler` before making predictions.

## 📊 Dataset

The project uses the **Breast Cancer Wisconsin Diagnostic Dataset**.

The model uses 30 numerical tumor features, including:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Concave Points
- Symmetry
- Fractal Dimension

These measurements are provided as **Mean, Standard Error (SE), and Worst** values.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SVM
- StandardScaler
- Joblib
- Streamlit
- Google Colab
- GitHub

## 📁 Project Structure
Breast_Cancer_SVM/
│
├── app.py
├── SVM.ipynb
├── svm_model.pkl
├── scaler.pkl
├── cancer_image.png
├── requirements.txt
└── README.md

## 🚀 How to Run

1) Clone the repository:
git clone https://github.com/harekrishna10/breastcancerwebsite.git

2) Go to the project folder:
cd breastcancerwebsite

3) Install the required libraries:
pip install -r requirements.txt

4) Run the Streamlit application:
streamlit run app.py


## ⚠️ Disclaimer

This application is developed for educational purposes only. It is not a medical diagnostic tool and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

## 👨‍💻 Author

Harekrishna Chaudhary

BIM / Information Technology Student / Machine Learning / Data Science
