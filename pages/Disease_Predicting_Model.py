import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from PIL import Image
import base64

@st.cache_data
def load_data():
    return pd.read_csv("Datasets/Training.csv")
training_data = load_data()
testing_data=pd.read_csv('Datasets/Testing.csv')

# Define the symptom list (based on the dataset columns)
symptoms = training_data.columns[:-1].tolist()  # All columns except the last one (disease)
    
# Prepare the data
X_train = training_data[symptoms]
y_train = training_data["prognosis"]
    

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
    
 # Custom styling for title and subheadings
st.markdown(
        """
        <style>
        .model-title {   /* changing colour, fontsize, font of title */
            color: #fad4cf;
            font-size: 40px;
            font-weight: bold;
            font-family: "Times New Roman", Times, serif;
        }
        .subheading {      /* changing colour,fontsize of subheading  */
            color: white;
            font-size: 20px;
            margin-top: 20px;
        }
        .st-emotion-cache-10bdsh3 { /* changing colour of label symptoms */
            color: white;
            font-size: 15px;
        }

        .st-emotion-cache-6qob1r {
        background-color: #FFDDC1;  /* changing colour of sidebar */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Title with custom class
st.markdown('<div class="model-title">Disease Prediction System</div>', unsafe_allow_html=True)

# Instruction for the user
st.markdown('<div class="subheading">Select the symptoms you are experiencing from the list below:</div>', unsafe_allow_html=True)

selected_symptoms=st.multiselect(label='Symptoms',options=symptoms)

# Create input data for the model
input_data = np.zeros(len(symptoms))  # Start with all symptoms as 'not selected'


for symptom in selected_symptoms:
    index = symptoms.index(symptom)
    input_data[index] = 1  # Mark selected symptoms as 1

# Add some vertical space before the button
st.write("\n\n")

# Predict button
if st.button("Predict Disease"):
        if np.sum(input_data) == 0:
            st.markdown(
                f"""
                <div style="
                    background-color: #f5a98c;
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.01);
                    color: red;
                    font-size: 24px;
                    text-align: center;
                    font-weight: bold;">
                    Please select at least one symptom to make a prediction.
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            input_data = input_data.reshape(1, -1)  # Reshape for prediction
            predicted_disease = model.predict(input_data)  # Predict the encoded label
            st.markdown(
                f"""
                <div style="
                    background-color: #ebc775;
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                    color: #155724;
                    font-size: 24px;
                    text-align: center;
                    font-weight: bold;">
                    Predicted Disease: {predicted_disease[0]}
                </div>
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
add_bg_from_local('iStock-849243802-scaled.jpg')
