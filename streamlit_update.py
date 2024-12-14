# -*- coding: utf-8 -*-
import streamlit as st
import os
from dotenv import load_dotenv
from PIL import Image
import pickle
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Set up Groq client
client = Groq(api_key=api_key)
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# Streamlit app title and configuration
st.set_page_config(page_title="GrowBuddy")
st.title("I am Your GrowBuddy 🪴")
st.write("Your supportive guide for urban gardening. Let's cultivate a greener space together!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello there! How can I help you with gardening today?"}
    ]

# Chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert assistant for urban gardening, providing concise advice on plant care, "
            "soil preparation, organic pest control, and sustainable gardening practices. Keep responses short. "
            "If unsure, respond with 'I don’t have enough information.'"
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

# Define the chain
chain = prompt | llm

# Helper functions
def get_text_response(query):
    """Generate a text response based on user input."""
    st.session_state.messages.append({"role": "user", "content": query})
    response = chain.invoke({"messages": st.session_state.messages})
    answer = response.content
    st.session_state.messages.append({"role": "assistant", "content": answer})
    return answer

def get_image_response(image, question):
    """Generate a response for uploaded images (placeholder logic for Groq)."""
    combined_input = f"Image uploaded with question: {question}" if question else "Image uploaded. Please identify the plant."
    st.session_state.messages.append({"role": "user", "image": image, "content": question})
    response = chain.invoke({"messages": st.session_state.messages})
    answer = response.content
    st.session_state.messages.append({"role": "assistant", "content": answer})
    return answer

# Image processing helper functions
def extract_color_histogram(image):
    """Extract color histogram features for image classification."""
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def classify_image(uploaded_image):
    """Classify uploaded plant image using trained models."""
    try:
        # Load and display the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            st.error("Uploaded file is not a valid image.")
            return None

        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

        # Resize the image
        IMG_SIZE = 128  # Ensure this matches the size used in training
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        # Extract features
        features = extract_color_histogram(image)

        # Load scaler
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        features = scaler.transform([features])  # Scale features

        # Load LabelEncoder
        with open("label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)

        # Load models and make predictions
        models = {
            "Decision Tree": "decision_tree_model.pkl",
            #"Random Forest": "random_forest_model.pkl",
            #"SVM": "svm_model.pkl",
            "K-Nearest Neighbors": "knn_model.pkl",
            "Logistic Regression": "logistic_regression_model.pkl"
        }

        predictions = {}
        for model_name, model_file in models.items():
            with open(model_file, "rb") as f:
                model = pickle.load(f)
                pred = model.predict(features)  # Numeric label prediction
                pred_label = label_encoder.inverse_transform(pred)[0]  # Map numeric label back to class name
                predictions[model_name] = pred_label

        return predictions
    except Exception as e:
        st.error(f"An error occurred during classification: {e}")
        return None

# Chat interaction
user_prompt = st.chat_input("Type your question below:")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            answer = get_text_response(user_prompt)
            st.write(answer)

uploaded_image = st.file_uploader("Upload an image (optional):", type=["png", "jpg", "jpeg"])
if uploaded_image:
    predictions = classify_image(uploaded_image)
    if predictions:
        st.subheader("Model Predictions")
        for model_name, pred_label in predictions.items():
            st.write(f"{model_name}: {pred_label}")
    question = st.text_input("Ask about the uploaded image:")
    if question:
        with st.chat_message("assistant"):
            with st.spinner("Processing your image query..."):
                answer = get_image_response(uploaded_image, question)
                st.write(answer)

# Disclaimer
st.markdown('<p style="font-size:10px;">GrowBuddy may make mistakes. Always consult experts. Only necessary data is collected to provide accurate responses.</p>', unsafe_allow_html=True)
