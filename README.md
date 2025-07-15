# 🔎 CardioPredict AI Pro

An enhanced AI-powered web application for real-time cardiovascular risk assessment and angina prediction, combining interactive UI elements, explainable models, and rich visualizations.

## 🌌 Overview

**CardioPredict AI Pro** leverages a PyCaret-trained LightGBM model to estimate patient risk of angina based on a combination of clinical, demographic, and biochemical indicators. It includes modern UI theming, multilingual support, feature explanations, historical projections, and population comparisons.

## 📊 Features

- Real-time angina risk prediction
- Multilingual interface (English, Spanish, French)
- Animated gauge and radar charts for health metrics
- Feature attribution and SHAP-style visualizations
- Risk trend timeline and lifestyle/medication impact simulations
- Patient history tracking and comparison tools
- Enhanced report generation with dynamic clinical recommendations
- Light/Dark theme toggle and accessibility options

## 📊 Input Parameters

Accepts 35+ patient features:

- **Demographics**: Age, Sex, Ethnicity
- **Vitals**: Blood Pressure, Heart Rate, Chest Pain
- **Lifestyle**: Smoking Status, Physical Activity, BMI
- **Lab Values**: Cholesterol, Glucose, HbA1c, Creatinine, Hemoglobin, etc.
- **Medical History**: Diabetes, Hypertension, Steroid Use, Family CHD

## 🌐 Output

- Risk Classification: LOW / MODERATE / HIGH
- Predicted probability with confidence range
- Feature-level contributions to prediction
- Clinical report with recommendations based on risk level

## 🎓 Visualizations

- Enhanced gauge meter
- Radar chart comparing patient vs optimal range
- Heatmap vs population percentiles
- SHAP-style bar chart for feature impacts
- Timeline projection with and without interventions

## 🚀 Getting Started

1. Clone/download the repository
2. Ensure Python 3.8+ is installed
3. Install dependencies:

```bash
pip install streamlit pandas pycaret pillow plotly seaborn matplotlib scikit-learn requests
```

4. Run the app:

```bash
streamlit run predict_angina_app.py
```

## 📁 Included Files

- `predict_angina_app.py`: Main app file
- `All_Variables_Model_LightGBM.pkl`: ML model (required)
- Optional assets: Lottie animations, images, audio modules

## 🧠 Technical Details

- **Model**: LightGBM Classifier via PyCaret
- **Interface**: Streamlit with Plotly and custom CSS animations
- **Logic**: Enhanced visual insights, undo/redo state, speech-to-text support

## 📌 Disclaimer

This tool is intended for **research and educational purposes only** and does not constitute medical advice. Always consult a healthcare provider for clinical decisions.

## 📩 Contact

For inquiries, collaborations, or feedback:

**ALRUBAYYI ABDULAZIZ**\
[Abdulaziz.Alrubayyi@warwick.ac.uk](mailto:Abdulaziz.Alrubayyi@warwick.ac.uk)
