import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import base64

# Custom styling for title and subheadings
st.markdown(
        """
        <style>
        .model-title {   /* changing colour, fontsize, font of title */
            color: #fad4cf;
            font-size: 40px;
            font-weight: bold;
        }
        .subheading {      /* changing colour,fontsize of subheading  */
            color: white;
            font-size: 30px;
            margin-top: 20px;
        }

        .st-emotion-cache-6qob1r {
        background-color: #FFDDC1;  /* changing colour of sidebar */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Title with custom class
st.markdown('<div class="model-title">All About Dataset That We Have Used</div>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("Datasets\Training.csv")
df = load_data()
st.markdown('<div class="subheading">First 5 Rows of Dataset</div>', unsafe_allow_html=True)
st.write(df.head())
st.markdown('<div class="subheading">Last 5 Rows of Dataset</div>', unsafe_allow_html=True)
st.write(df.tail())
st.markdown('<div class="subheading">Descriptive Statistics of Dataset</div>', unsafe_allow_html=True)
st.write(df.describe())
st.markdown('<div class="subheading">Columns in dataset</div>', unsafe_allow_html=True)
st.write(df.columns)
st.markdown('<div class="subheading">Total Rows and Columns in dataset</div>', unsafe_allow_html=True)
st.write(df.shape)
st.markdown('<div class="subheading">Count The occurrences of each unique value in Prognosis column</div>', unsafe_allow_html=True)
st.write(df['prognosis'].value_counts())

#adding background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
            background-size: cover
            
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
add_bg_from_local('Screenshot 2024-10-30 221031.png')