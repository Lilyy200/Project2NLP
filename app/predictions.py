#\utilitaires\predictions.py
import joblib
from model_files import model_files
import tensorflow as tf

# Function to load the model based on the file path
def load_model(model_name):
    model_path = model_files.get(model_name)
    
    if model_path:
        # Load the model based on its extension
        if model_path.endswith(".h5"):  # Keras model (e.g., BERT or custom embedding)
            print(f"Loading Keras model: {model_name}")
            return tf.keras.models.load_model(model_path)
        elif model_path.endswith(".pkl"):  # scikit-learn model
            print(f"Loading scikit-learn model: {model_name}")
            return joblib.load(model_path)
        elif model_path.endswith(".model"):  # Other formats like Word2Vec
            print(f"Loading model: {model_name}")
            return joblib.load(model_path)  # Adjust if you're using a different format for Word2Vec
        else:
            raise ValueError(f"Unsupported model file format for {model_name}")
    else:
        raise ValueError(f"Model {model_name} not found.")
    