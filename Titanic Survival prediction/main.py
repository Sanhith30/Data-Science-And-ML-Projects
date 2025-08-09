import streamlit as st
import math
import joblib  

model = joblib.load("model.pkl")

def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://static1.moviewebimages.com/wordpress/wp-content/uploads/article/3bB2UnWTUkslTngQMkwRZbO3qdYkhD.jpg?q=70&fit=contain&w=1200&h=628&dpr=1");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

st.header("Titanic Survival Prediction !!")

col1, col2, col3 = st.columns([1,1,1])
with col1:
    Pclass = st.selectbox("Class of Passenger", ("Premiere", "Executive", "Economy"))
with col2:
    Sex = st.selectbox("Gender", ("Male", "Female"))
with col3:
    Age = st.number_input("Age of passenger", min_value=0, max_value=120)

col4, col5 = st.columns([1,1])
with col4:
    SibSp = st.number_input("Siblings/Spouses", min_value=0)
with col5:
    Parch = st.number_input("Parents/Children", min_value=0)

col7, col8 = st.columns([1,1])
with col7:
    Fare = st.number_input("Fare of Journey", min_value=0.0, format="%.2f")
with col8:
    Embarked = st.selectbox("Picking Point", ("Cherbourg", "Queenstown", "Southampton"))

if st.button("Predict"):
    pclass = 1
    if Pclass == "Economy":
        pclass = 3
    elif Pclass == "Executive":
        pclass = 2

    gender = 0
    if Sex == "Female":
        gender = 1

    age = math.ceil(Age)
    sibsp = math.ceil(SibSp)
    parch = math.ceil(Parch)
    fare = math.ceil(Fare)

    embarked = 0
    if Embarked == "Cherbourg":
        embarked = 1
    elif Embarked == "Queenstown":
        embarked = 2

    result = model.predict([[pclass, gender, age, sibsp, parch, fare, embarked]])
    output_labels = {
        1: "The passenger will Survive",
        0: "The passenger will not survive"
    }
    st.markdown(f"## {output_labels[result[0]]}")
