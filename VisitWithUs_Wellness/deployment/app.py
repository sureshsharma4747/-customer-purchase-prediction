import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="sureshsharma4747/Customer-Purchase-Model", filename="best_customer_purchase_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Prediction App")
st.write("""
This application predicts the likelihood of a customer purchasing based on its operational parameters.
Please enter the sensor and configuration data below to get a prediction.
""")

# User input
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=100, value=15)
number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
number_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2)
preferred_property_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
number_of_trips = st.number_input("Number of Trips", min_value=0, max_value=20, value=3)
passport = st.selectbox("Passport", [0, 1]) # 0 = No, 1 = Yes
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
own_car = st.selectbox("Own Car", [0, 1]) # 0 = No, 1 = Yes
number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=200000, value=50000, step=1000)

# Assemble input into DataFrame
input_data = pd.DataFrame([{ "Age": age,
                            "TypeofContact": typeof_contact,
                             "Occupation": occupation,
                             "Gender": gender,
                             "ProductPitched": product_pitched,
                             "MaritalStatus": marital_status,
                             "Designation": designation,
                             "CityTier": city_tier,
                             "DurationOfPitch": duration_of_pitch,
                             "NumberOfPersonVisiting": number_of_person_visiting,
                             "NumberOfFollowups": number_of_followups,
                             "PreferredPropertyStar": preferred_property_star,
                             "NumberOfTrips": number_of_trips,
                             "Passport": passport,
                             "PitchSatisfactionScore": pitch_satisfaction_score,
                             "OwnCar": own_car,
                             "NumberOfChildrenVisiting": number_of_children_visiting,
                             "MonthlyIncome": monthly_income
}])


if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Will Purchase" if prediction == 1 else "Will Not Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
