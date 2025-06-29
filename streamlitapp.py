import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
# Load the Iris dataset and train the model
X, y = load_iris(return_X_y=True)
clf = LogisticRegression(random_state=0, max_iter=200).fit(X, y)
class_names = ['Setosa', 'Versicolor', 'Virginica']
st.title("Iris Flower Prediction App")
st.write("This app predicts the species of Iris flower based on its features.")
sepal_length = st.slider("Sepal Length (cm)", 4.3, 7.9, 5.8)
sepal_width  = st.slider("Sepal Width (cm)", 2.0, 4.4, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 6.9, 4.3)
petal_width  = st.slider("Petal Width (cm)", 0.1, 2.5, 1.5)
if st.button("Predict"):
    input_data = [sepal_length, sepal_width, petal_length, petal_width]
    result = clf.predict([input_data])[0]
    st.success(f"The predicted species is: **{class_names[result]}**")
    

    

