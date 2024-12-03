from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import confusion_matrix
import streamlit as st
import pandas as pd
import numpy as np

import json
import joblib

st.title("Heart Disease Prediction Model")

models = (
    "XGBoost_model.joblib",
    "RandomForest_model.joblib"
)
selected_model = st.selectbox("Choose a model to evaluate your infos", options=models, index=0)

@st.cache_resource
def load_model(model_path: str):
    try:
        model = joblib.load(model_path)
        st.success(f"Model '{model_path}' loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load model '{model_path}'. Error: {e}")
        return None

# Load the model based on user selection
sl_model = load_model(selected_model)

# Display model loading status
if sl_model:
    st.text("Model is ready for prediction.")
else:
    st.text("Please select a valid model.")
    

def run_webapp(model):
    cnt = 0
    st.markdown(
        """
        <style>
        .stImage {
            display: flex;
            justify-content: flex-end;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Heart Attack Prediction")
        st.subheader("Do you have a healthy heart?")

    with col2:
        st.image("heart.jpg", width=150)


    #cp, trtrbps, exng, slp, caa, thall
    with st.form("heart_attack_form"):
        age = st.slider("Your Age [age]", 18, 100, value=30)
        sex = st.selectbox("Your Gender [sex]", [0, 1], format_func=lambda x: ["Women", "Men"][x], index=1)
        chol = st.slider("Your Cholesterol Measurement[mg/dl] [chol]", 200, 600, value= 400)
        st.text("Is Your Fasting Blood Sugar Greater Than 120 mg/dl? [fbs]")
        fbs = st.checkbox("Yes it is")
        
        restecg = st.selectbox("Your Resting ECG Results [restecg]", [0, 1, 2],
                               format_func=lambda x: ["normal",
                                                    "having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)",
                                                    "showing probable or definite left ventricular hypertrophy by Estes’ criteria"][x],
                               index=0
        )

        oldpeak = st.slider("Your ST depression induced by exercise relative to rest [oldpeak]", 0.0, 6.0, value=3.0)
        thalach = st.slider("Your Maximum Heart Rate [thalach]", 70, 200, value=145)

        cp = st.selectbox("------ Your Chest Pain Type [cp] -----", [0, 1, 2, 3],
                          format_func=lambda x: ["asymptomatic (tünetmentes)", "atypical angina", "typical angina (H)", "non-anginal pain"][x], index=0)
        trtbps = st.slider("------ Your Resting Blood Pressure [mm/Hg] [trtbps] --------", 90, 200, value=120, step=5)
        st.text("------- Have You Ever Experienced Angina When Exercising? [exng] --------")
        exng = st.checkbox("Yes I have")
        slp = st.selectbox("------- The slope of the peak exercise ST segment [slp] -------", [0, 1, 2], format_func=lambda x: ["downsloping (L)", "flat", "upsloping (H)"][x]
        )
        caa = st.slider("-------- The number of major vessels (0–4) [caa] --------", 0, 4, value=1, step=1)
        thall = st.selectbox("-------- A blood disorder called thalassemia [thall] --------", [0, 1, 2],
                            format_func=lambda x: ["normal blood flow (L)",
                            "fixed defect (no blood flow in some part of the heart) (H)",
                            "reversible defect (a blood flow is observed but it is not normal)"][x]
        ) # add +1 at input data
        #['cp', 'trtbps', 'exng', 'slp', 'caa', 'thall']
        submitted = st.form_submit_button("Submit")

        if submitted:

#['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall']
            input_data = {
                "age": age,
                "sex": sex,
                "cp": cp,
                "trtbps": trtbps,
                "chol": chol,
                "fbs": fbs,
                "restecg":restecg,
                "thalachh": thalach,
                "exng": exng,
                "oldpeak": oldpeak,
                "slp": slp,
                "caa": caa,
                "thall": thall+1
            }
            input_df = pd.DataFrame([input_data])

            scaler = joblib.load("scaler_cp_trtbps_exng_slp_caa_thall.pkl")

            scaled_array = scaler.transform(input_df)
            input_df_scaled = pd.DataFrame(scaled_array, columns=input_df.columns)

            print("Input DataFrame Columns:", input_df_scaled.columns)
            print("input DataFrame:", input_df_scaled, sep="\n")


            required_cols = ['cp', 'trtbps', 'exng', 'slp', 'caa', 'thall']
            try:
                input_df_scaled = input_df_scaled[required_cols]
            except KeyError as e:
                st.error(f"Missing column: {e}")
                st.stop()



            prediction = model.predict(input_df_scaled)[0]

            print("pred", prediction)

            prediction_label = "High risk" if prediction == 1 else "Low risk"
            st.write(f"### Prediction: {prediction_label}")

            input_df["prediction"] = prediction_label
            # Debug
            print("Updated Input DataFrame:\n", input_df, sep="\n")
            if "results_df" not in st.session_state:
                st.session_state["results_df"] = pd.DataFrame(columns=input_df.columns)

            st.session_state["results_df"] = pd.concat([st.session_state["results_df"], input_df], ignore_index=True)

            st.write("### Recorded Inputs and Predictions")
            st.dataframe(st.session_state["results_df"])

            #cp, rtrbps, exng, slp, caa, thall

            # print(f"Age: {age}")
            # print(f"Sex: {sex}")
            # print(f"Fasting Blood Sugar: {fbs}")
            # print(f"Cholesterol: {chol}")
            # print(f"Resting ECG: {restecg}")
            # print(f"Maximum Heart Rate: {thalach}")
            # print(f"Chest Pain Type: {cp}")
            # print(f"ST Depression: {oldpeak}")
            print(f"Resting Blood Pressure: {trtbps}")
            print(f"Exercise Induced Angina: {exng}")
            print(f"Slope: {slp}")
            print(f"Number of Major Vessels: {caa}")
            print(f"Thalassemia: {thall}")

if __name__ == "__main__":
    run_webapp(sl_model)
