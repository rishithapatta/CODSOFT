import streamlit as st
import pickle
import re
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text
st.title("Movie Genre Classification")
st.write("Enter a movie plot to predict its genre")
user_input = st.text_area("Movie Plot")
if st.button("Predict Genre"):
    if user_input:
        clean = clean_text(user_input)
        vector = tfidf.transform([clean])
        prediction = model.predict(vector)

        st.success(f"Predicted Genre: {prediction[0]}")
    else:
        st.warning("Please enter a plot")