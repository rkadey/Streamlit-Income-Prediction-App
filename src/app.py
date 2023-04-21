import pickle
import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image

st.set_page_config(
    page_title="Prediction",
    page_icon="🎈",
)


@st.cache_data
def load_model():
    with open('ml_pipeline.pkl', 'rb') as file:
        pipeline = pickle.load(file)
    return pipeline

# Load ML components
pipeline = load_model()
data = pipeline["data"]
classifier = pipeline["pipeline"]


# App Interface
st.write("# Welcome to Income Prediction App! 🎈")
image = Image.open("pexels-andrea-piacquadio-3823487.jpg")
st.image(image )
st.sidebar.markdown(
 """The task is to predict whether a given adult makes more than 
$50,000 a year or not.
"""
)
form = st.form("my_form", clear_on_submit=True)
with form:

    demancator = st.columns((1, 1))

    input = {"Age":[demancator[0].slider("Age",
                                           int(data.Age.min()), 
                                           int(data.Age.max()), 20)],
    "workclass":[demancator[0].selectbox("Workclass", 
                                           data.workclass.unique())],
    "fnlwgt":[demancator[0].slider("Final Weight", 
                                     int(data.fnlwgt.min()), 
                                     int(data.fnlwgt.max()), 50000)],
    "education":[demancator[0].selectbox("Highest level of Education",
                                            data.education.unique())],
    "education_num":[demancator[0].slider("Highest level of education achieved in numerical form", 
                                           int(data.education_num.min()), 
                                           int(data.education_num.max()), 5)],
    "marital_status":[demancator[0].selectbox("Marital Status", 
                                                data["marital_status"].unique())],
    "occupation":[demancator[0].selectbox("Your Occupation", 
                                            data.occupation.unique())],
    "relationship":[demancator[1].selectbox("Relationship", 
                                              data.relationship.unique())],
    "race":[demancator[1].selectbox("Race", 
                                      data.race.unique())],
    "sex":[demancator[1].radio("Gender", 
                                 data.sex.unique())],
    "capital_gain":[demancator[1].slider("Capital Gains", 
                                           int(data.capital_gain.min()), 
                                           int(data.capital_gain.max()), 500)],
    "capital_loss":[demancator[1].slider("Capital Loss", 
                                           int(data.capital_loss.min()), 
                                           int(data.capital_gain.max()), 500)],
    "hours_per_week":[demancator[1].slider("Hours per Week", 
                                            int(data.hours_per_week.min()), 
                                            int(data.hours_per_week.max()), 20)],
    "native_country":[demancator[1].selectbox("Native Country", 
                                                data["native_country"].unique())],
    "age_group":[demancator[0].selectbox("Age Group", 
                                           data["age_group"].unique())]
    }
    

    prediction = st.form_submit_button("Predict")
    data_input = pd.DataFrame(input)

    if prediction:
        try:
            st.balloons()
            prediction_output = classifier.predict(data_input)
            if prediction_output == [0]: 
                st.header(f"Predicted income is less than or equal to $50,000")
            else:
                st.header(f"Predicted income is greater than  $50,000")
        except Exception as e:
            print(e)