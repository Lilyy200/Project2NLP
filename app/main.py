import streamlit as st
import joblib
import os
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub

from predictions import load_model  # This imports your model loading function
from data_cleaning import remove_french_stopwords, load_tokenizer, preprocess_input_text, remove_english_stopwords
from model_files import model_files

def run():
    # Title and Subheader
    st.title("Model Prediction")
    st.subheader("Choose the model for prediction")

    # Model selection (list all available models)
    model_choice = st.selectbox("Select Model", list(model_files.keys()))
    
    # Input box for user to enter a review
    user_input = st.text_area("Enter your review:", height=200)

    # Button to trigger prediction
    if st.button('Predict'):
        if user_input:
            # Load the tokenizer
            tokenizer = load_tokenizer()
            
            # Check if cleaning is necessary (skip for BERT)
            if model_choice in ["Linear Regression", "SVM", "Random Forest", "AdaBoost", "Gradient Boosting", "my_embedding", "pretrained embedding"]:
                # Preprocess the text if using classical models
                cleaned_text = remove_french_stopwords(user_input)
                if model_choice in ["my_embedding", "pretrained embedding"]:
                    # Load the tokenizer for padding the input text
                    cleaned_text = preprocess_input_text(cleaned_text, tokenizer)
            elif model_choice in ["use","sentence_bert"]:
                # For BERT and other models, skip preprocessing
                cleaned_text = user_input

            else:
                # Skip preprocessing for other models
                cleaned_text = user_input

            # Load the selected model
            model = load_model(model_choice)

            if model_choice == "use":
                # For the pretrained embedding model (Word2Vec model) or similar
                embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

                # Ensure user_input is passed as a list of one item (since USE expects a batch)
                cleaned_text_list = [cleaned_text]  # Wrap the single string in a list

                # Encode the text data using USE
                user_input_encoded = embed(cleaned_text_list)  # This will return a tensor of shape [1, n_features]

                # Convert the tensor to a numpy array and flatten it to 1D
                user_input_encoded = user_input_encoded.numpy().flatten()

                # Reshape to 2D array (1 sample, n_features) for prediction
                cleaned_text = user_input_encoded.reshape(1, -1)

            elif model_choice == "sentence_bert":
                from sentence_transformers import SentenceTransformer

                # Load the Sentence-BERT model
                sentence_bert = SentenceTransformer('paraphrase-MiniLM-L6-v2')

                # Encode the user input into embeddings
                cleaned_text_embedding = sentence_bert.encode([cleaned_text])  # This returns a 2D array (1, embedding_size)
                

                # Use the embeddings as input for your regression/classification model
                prediction = model.predict(cleaned_text_embedding)

            else:
                # Predict using the loaded model for other model choices
                prediction = model.predict([cleaned_text])

            # Show the prediction result
            st.write(f"Prediction: {prediction}")
        else:
            st.error("Please enter a review for prediction.")
