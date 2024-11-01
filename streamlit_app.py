import streamlit as st


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

# Title and subtitle
col1, col2 = st.columns([3, 1])
# Title and subtitle in the left column
with col1:
    st.title("Heart Attack Prediction")
    st.subheader("Do you have a healthy heart?")

# Image in the right column
with col2:
    st.image("heart.jpg", width=150)

# Form for user input
with st.form("heart_attack_form"):
    age = st.slider("Your Age", 18, 100, value=60,)
    sex = st.selectbox("Your Gender", [0, 1], format_func=lambda x: ["Women", "Men"][x], index=1)
    cp = st.selectbox("Your Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: ["asymptomatic", "atypical angina", "non-anginal pain", "typical angina"][x])
    trestbps = st.slider("Your Resting Blood Pressure [mm/Hg]", 90, 200, value=145)
    chol = st.slider("Your Cholesterol Measurement[mg/dl]", 200, 600, value= 400)
    st.text("Is Your Fasting Blood Sugar Greater Than 120 mg/dl?")
    fbs = st.checkbox("Yes it is")
    restecg = st.selectbox("Your Resting ECG Results", [0, 1, 2], format_func=lambda x: ["showing probable or definite left ventricular hypertrophy by Estes’ criteria", "normal", "having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)"][x])
    thalach = st.slider("Your Maximum Heart Rate", 70, 200, value=145)
    st.text("Have You Ever Experienced Angina When Exercising?")
    exang = st.checkbox("Yes I have")
    oldpeak = st.slider("Your ST depression induced by exercise relative to rest", 0.0, 6.0, value=3.0)
    slope = st.selectbox("The slope of the peak exercise ST segment", [0, 1, 2], format_func=lambda x: ["downsloping", "flat", "upsloping"][x])
    ca = st.slider("The number of major vessels (0–3)", 0.0, 3.0, value=1.5, step=0.1)
    thal = st.selectbox("A blood disorder called thalassemia", [0, 1, 2], format_func=lambda x: ["fixed defect (no blood flow in some part of the heart)", "normal blood flow", "reversible defect (a blood flow is observed but it is not normal)"][x])

    # Submit button
    submitted = st.form_submit_button("Submit")

    if submitted:
        st.write("Form submitted successfully!")
        
        #model

        print(f"Age: {age}")
        print(f"Sex: {sex}")
        print(f"Chest Pain Type: {cp}")
        print(f"Resting Blood Pressure: {trestbps}")
        print(f"Cholesterol: {chol}")
        print(f"Fasting Blood Sugar: {fbs}")
        print(f"Resting ECG: {restecg}")
        print(f"Maximum Heart Rate: {thalach}")
        print(f"Exercise Induced Angina: {exang}")
        print(f"ST Depression: {oldpeak}")
        print(f"Slope: {slope}")
        print(f"Number of Major Vessels: {ca}")
        print(f"Thalassemia: {thal}")


# Run the app
if __name__ == "__main__":
    pass
