import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import spacy
import numpy as np
import os

def loadObject(file_path):        
    file_path = file_path + ".pickle"

    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        return obj
    except Exception as e:
        return None

@st.cache_data
def load():
    # Load cached objects and resources

    tokenizer = loadObject(r'simple_chatbot_code\tokenizer')
    le = loadObject(r'simple_chatbot_code\label_encoder')
    tags_answers = loadObject(r'simple_chatbot_code\tags_answers')
    max_len = loadObject(r'simple_chatbot_code\MAX_LEN')

    model = load_model(r'simple_chatbot_code\simple_chatbot_val_model.h5')

    nlp = spacy.load(r'en_core_web_sm')

    return tokenizer, le, tags_answers, model, nlp, max_len

def cleanPattern(msg, nlp):
    pat_char = re.compile(r'[^A-Za-z]')
    pat_spaces = re.compile(r'\s+')

    msg = str(msg).lower()
    msg = msg.strip()
    msg = re.sub(pat_char,' ', msg)
    msg = re.sub(pat_spaces,' ', msg)
    
    tokens = nlp(msg)
    lemma = [token.lemma_ for token in tokens]

    return ' '.join(lemma)

def predict(msg):
    tokenizer, le, tags_answers, model, nlp, max_len = load()

    msg = cleanPattern(msg, nlp)

    sequences = tokenizer.texts_to_sequences([msg])
    padded_sequences = pad_sequences(sequences, padding='post', maxlen=max_len)

    predictino_prob = model.predict(padded_sequences)

    tag_index = np.argmax(predictino_prob , axis=1 )
    
    tag = le.inverse_transform(tag_index)[0]

    all_responses = tags_answers[tag]

    random_response = np.random.choice(all_responses)

    return random_response
    
def app():
    st.set_page_config(
        page_title="AI Chatbot",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 AI Chatbot")
    st.markdown("DEPI Graduation Proj")
    st.markdown("🔗 The journey of building an intelligent chatbot combines creativity, data preprocessing, and cutting-edge technology. Through a mix of data exploration, NLP techniques, and deep learning models, ")

    st.markdown("Welcome! Ask me anything or type a message below to start chatting.")

    # Load resources with spinner
    with st.spinner("Loading chatbot resources..."):
        load()

    # Initialize chat messages
    if "msg" not in st.session_state:
        st.session_state["msg"] = []

    # User input field
    user_input = st.chat_input("Type your message here...")
    if user_input:
        # Append user message and bot response
        st.session_state["msg"].append((0, user_input))
        bot_response = predict(user_input)
        st.session_state["msg"].append((1, bot_response))

    # Display chat messages
    for sender, message in st.session_state["msg"]:
        if sender == 0:
            with st.chat_message("👱"):
                st.write( f"{message}")
        else:
            with st.chat_message("🤖"):
                st.write(f"{message}")

    # Sidebar for additional options
    with st.sidebar:
        st.header("⚙️ Settings")
        if st.button("Clear Chat"):
            st.session_state["msg"] = []
            st.rerun()

if __name__ == "__main__":
    app()