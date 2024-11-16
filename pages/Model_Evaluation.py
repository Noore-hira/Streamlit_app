import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from PIL import Image
import base64

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
            font-weight: bold;
        }
        .st-emotion-cache-183lzff{
            color: white;
            font-size: 14px;
        }
        .st-emotion-cache-6qob1r {
        background-color: #FFDDC1;  /* changing colour of sidebar */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Title with custom class
st.markdown('<div class="model-title">Evaluating Model Performance</div>', unsafe_allow_html=True)

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

#Evaluate the model
# Separate features and target variable in test data
X_test = testing_data[symptoms]
y_test = testing_data['prognosis']

# Make predictions on the test set
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.write('\n')
st.markdown(f'<div class="subheading"> Model Accuracy : {accuracy:.2f}', unsafe_allow_html=True)
st.markdown(f'<div class="subheading"> Classification Report', unsafe_allow_html=True)
st.text(report)

 # Confusion Matrix
st.markdown(f'<div class="subheading"> Confusion Matrix', unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=y_test.unique(), yticklabels=y_test.unique())
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)
