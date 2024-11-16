import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64

@st.cache_data
def load_data():
    return pd.read_csv("Datasets/Training.csv")
df = load_data()
st.markdown(
        """
        <style>
        .st-emotion-cache-asc41u h1 {   /* changing colour, fontsize, font of title */
            color: #fad4cf;
            font-size: 40px;
            font-weight: bold;
        }
        h2 {      /* changing colour,fontsize of subheading  */
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


st.title('Data Visualization')

st.write("## Target Variable Distribution")
plt.figure(figsize=(10, 5))
sns.countplot(x=df["prognosis"],palette='rocket')
plt.xticks(rotation=90)
st.pyplot(plt)

st.write("## Symptom Correlations")
# Compute correlation matrix and plot heatmap
plt.figure(figsize=(12, 10))
symptom_corr = df.iloc[:, :-1].corr()  # Exclude the target column
sns.heatmap(symptom_corr, cmap="coolwarm", center=0, annot=False)
st.pyplot(plt)

st.write("## Symptom Frequency")
symptom_counts = df.iloc[:, :-1].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 30))
sns.barplot(x=symptom_counts.values, y=symptom_counts.index,palette='dark:salmon_r')
st.pyplot(plt)

# 4. Symptoms vs. Disease Heatmap
st.write("## Symptom Presence Across Diseases")
plt.figure(figsize=(14, 10))
disease_symptom_matrix = df.groupby('prognosis').sum()
sns.heatmap(disease_symptom_matrix, cmap='Spectral', annot=False)
plt.xlabel("Symptoms")
plt.ylabel("Diseases")
plt.show()
st.pyplot(plt)
