import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import base64


st.set_page_config(page_title='Homepage')
st.title('Introduction')

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


# Set the background color for the sidebar
st.markdown(
    """
    <style>
    .st-emotion-cache-6qob1r {
        background-color: #FFDDC1;  /* Light peach color */
    }

    .st-emotion-cache-asc41u h1 {   /* changing colour, fontsize, font of title */
            color: #fad4cf;
            font-size: 40px;
            font-weight: bold;
        }
    h3{      /* changing colour,fontsize of subheading  */
            color: white;
            font-size: 30px;
            margin-top: 20px;
            font-weight: bold;
        }
    .st-emotion-cache-1bixfc7{
            color: rgb(49, 51, 63);
            color: rgb(215, 217, 225);
            color: rgb(217, 219, 234);
            color-scheme: light;
            color: rgb(217, 219, 234);
            font-size: 30px;
            font-weight: bold;}
    .st-emotion-cache-1r4qj8v{
            color: rgb(49, 51, 63);
            color: rgb(215, 217, 225);
            color: rgb(217, 219, 234);
            color-scheme: light;
            color: rgb(217, 219, 234);
            font-size: 20px;
            font-weight: bold;
        }
    p, ol, ul, dl {
           font-size: 1rem;
           font-size: 1.1rem;
           margin: 0px 0px 1rem;
           padding: 0px;
           font-size: 1.1rem;
           font-weight: bold;
           font-weight: 400;
           font-weight: bold;
}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('''Welcome to the Disease Prediction System! This application uses a machine-learning model trained on medical symptom data to help predict potential diseases based on the symptoms selected by the user. Built using Streamlit and a Random Forest classifier, the app offers an intuitive, multi-page interface to explore data, visualize disease distributions, predict diseases, and evaluate the model's performance. This tool is designed for educational and informational purposes, offering a preview into symptom-based disease prediction.
            ''')
st.subheader('How to Use the Disease Prediction System')
st.markdown('''1. Select the Dataset page to review the data.
2. Explore the Data Visualization page to understand disease distribution within the training data.
3. Go to the Model page, select symptoms you experience, and click Predict Disease for a prediction.
4. Finally, visit Model Evaluation to review the model’s performance metrics.''')

st.write('Note: This application is intended solely for educational and informational purposes. The disease predictions provided here are based on a machine-learning model trained on a specific dataset and may not accurately reflect individual medical conditions. If you are experiencing symptoms or are concerned about your health, please consult a healthcare professional or doctor. Self-diagnosis and online tools should not be substituted for professional medical advice, diagnosis, or treatment.')