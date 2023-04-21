import streamlit as st
import pandas as pd
import os
import pickle
import sklearn 
import numpy as np
from sklearn.preprocessing import OneHotEncoder


# loading the trained model 
def load_model():
    with open('steps', 'rb') as file:
        data = pickle.load(file)
    return data


data = load_model()
decision_tree = data["model"]
#ohe_encoding = data["encoder"]

def date_transform(df,date):
    df["date"] = pd.to_datetime(df["date"])
    df["Year"] = df['date'].dt.year
    df["Month"] = df['date'].dt.month
    df["Week"] = df['date'].dt.isocalendar().week
    df["Day"] = df['date'].dt.day
    df['quarter'] = df['date'].dt.quarter
    df['week_of_year'] = df['date'].dt.isocalendar().week
    return df

# first line after the importation section
st.set_page_config(page_title="Sales predictor app", layout="centered")
DIRPATH = os.path.dirname(os.path.realpath(__file__))

@st.cache_resource()  # stop the hot-reload to the function just bellow

def setup(df):
    "Setup the required elements like files, models, global variables, etc"
    pd.DataFrame(
        dict(
            date=[],
            onpromotion=[],
            store_cluster=[],
            family=[],
            events=[],
            oil_price=[],    
        )
    ).to_csv(df, index=False)
    
df = os.path.join(DIRPATH, "df.csv")
setup(df)
 

#creating the interface
st.image("https://ultimahoraec.com/wp-content/uploads/2021/03/dt.common.streams.StreamServer-768x426.jpg")
st.title("Sales predictor app")

st.caption("This app predicts sales patterns of Cooperation Favorita over time in different stores in Ecuador.")

form = st.form(key="information", clear_on_submit=True)

with form:

    cols = st.columns((1, 1))
    date=cols[0].date_input("date")
    store_cluster = cols[1].number_input("store_cluster", min_value=1,max_value=17,step=1)
    family = cols[0].selectbox("family:", ["AUTOMOTIVE", "BEAUTY AND FASHION", "BEVERAGES AND LIQUOR", "FROZEN FOODS", "Grocery", "HOME AND KITCHEN", "HOME CARE AND GARDEN", "PET SUPPLIES", "SCHOOL AND OFFICE SUPPLIES"], index=2)
    events = cols[1].selectbox("events:", ["Holiday", "No holiday"])
    oil_price=st.slider("Enter the current oil price",min_value=1.00,max_value=100.00,step=0.1)
    onpromotion = st.slider("Enter the no of item on promotion",min_value=1.00,max_value=30.0,step=0.1)
    cols = st.columns(2)

    submitted = st.form_submit_button(label="Predict")
    

if submitted:
    pd.read_csv(df).append(
        dict(
            date=date,
            onpromotion=onpromotion,
            store_cluster=store_cluster,
            family=family,
            events=events,
            oil_price=oil_price,
            ),
        ignore_index=True,
    ).to_csv(df, index=False)
    df = pd.read_csv(df)

    new_df = date_transform(df,date)

    #Encoding on categorical columns.
    cat = ["family", "events"]
    encoder = OneHotEncoder(drop = "first", sparse=False)
    encoder.fit(new_df[cat])
    encoded_features =encoder.transform(new_df[cat])
    encoded_train = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out().tolist())
   
    new_df = pd.concat([df.reset_index(drop=True), encoded_train], axis=1)
    new_df = new_df.drop(["date","family", "events"], axis = 1)

    Prediction = decision_tree.predict(new_df)
    st.subheader(f"The prediction is {Prediction[0]}")

    st.balloons()

expander = st.expander("See all records")
with expander:
    df_new = pd.read_csv(df)
    st.dataframe(df_new)


