# -*- coding: utf-8 -*-
"""chatbot-streamlit.py"""

import streamlit as st
import joblib
import json
import random
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

model = joblib.load('best_rf_model_chatbot.pkl')

with open('KB.json', 'r') as file:
    data = json.load(file)

corpus = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        pattern = re.sub(r'\d+', '', pattern.lower())
        pattern = pattern.translate(str.maketrans('', '', string.punctuation))
        pattern = re.sub(r'\s+', ' ', pattern).strip()
        corpus.append(pattern)

vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)

def get_response(text):
    text = re.sub(r'\d+', '', text.lower())
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    vector = vectorizer.transform([text])
    tag = model.predict(vector)[0]
    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

st.set_page_config(page_title="Chatbot", layout="wide")
st.markdown("<h2 style='text-align:center'>Pandora - Therapeutic Chatbot</h2>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

chat_container = st.container()
user_input = st.chat_input("Say something...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    response = get_response(user_input)
    st.session_state.chat_history.append(("bot", response))

with chat_container:
    for sender, message in st.session_state.chat_history:
        with st.chat_message("user" if sender == "user" else "assistant"):
            st.markdown(message)
