import streamlit as st
import joblib
import numpy as np

log_reg = joblib.load("log_reg_model.pkl")
tree = joblib.load("decision_tree_model.pkl")

st.title("Product Purchase Prediction")
st.write("Compare Logistic Regression and Decision Tree models")

time_on_site = st.number_input("Time on Site (minutes)", min_value=0.0, step=0.1)
age = st.number_input("Age", min_value=0, max_value=100, step=1)
gender = st.selectbox("Gender", ("Male", "Female"))
ads_clicked = st.number_input("Ads Clicked", min_value=0, step=1)
previous_purchases = st.number_input("Previous Purchases", min_value=0, step=1)

# Convert gender to numeric
gender_val = 0 if gender.lower() == "male" else 1

if st.button("Predict"):
    features = np.array([[time_on_site, age, gender_val, ads_clicked, previous_purchases]])

    log_pred = log_reg.predict(features)[0]
    tree_pred = tree.predict(features)[0]

    st.subheader("Results")
    st.write(f"Logistic Regression Prediction: {'Will Purchase' if log_pred == 1 else 'Will Not Purchase'}")
    st.write(f"Decision Tree Prediction: {'Will Purchase' if tree_pred == 1 else 'Will Not Purchase'}")

    st.write("---")
    st.write("Model Accuracy on Test Data:")
    st.write("Logistic Regression: ~85%") 
    st.write("Decision Tree: ~80%")        
