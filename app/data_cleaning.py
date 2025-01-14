import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def preprocess_input_text(input_text, tokenizer, max_length=100):
    # Tokenize and pad the input text
    sequences = tokenizer.texts_to_sequences([input_text])  # Wrap input_text in a list
    padded_input = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_input

def load_tokenizer():
    with open('models/tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Function to remove French stopwords, including custom ones, without tokenizing
def remove_french_stopwords(text):
    # Default French stopwords
    stop_words_french = set(stopwords.words('french'))
    
    # Custom stopwords (hardcoded)
    custom_stopwords = [
        "faire", "fait", "a", "'", "’", "/", "lors", "l'un", "l'", "un", "je", "que", "le", "la", "et", "les", "de", "une", "du", "des", "dans", "ce", "cet", "cette",
        "avec", "sur", "pour", "tout", "mon", "mes", "ma", "à", "c'est", "j'ai", "eu", "chez", "sauf", "alors", "au", "aucuns", "aussi", "autre", "avant", "avec", "avoir", 
        "bon", "car", "ce", "cela", "ces", "ceux", "chaque", "ci", "comme", "comment", "dans", "des", "du", "dedans", "dehors", "depuis", "devrait", "doit", "donc", 
        "dos", "début", "elle", "elles", "en", "encore", "essai", "est", "et", "eu", "fait", "faites", "fois", "font", "hors", "ici", "il", "ils", "je", "juste", "la", 
        "le", "les", "leur", "là", "ma", "maintenant", "mais", "mes", "mien", "moins", "mon", "mot", "même", "ni", "nommés", "notre", "nous", "ou", "où", "par", 
        "parce", "pas", "peut", "peu", "plupart", "pour", "pourquoi", "quand", "que", "quel", "quelle", "quelles", "quels", "qui", "sa", "sans", "ses", "seulement", 
        "si", "sien", "son", "sont", "sous", "soyez", "sujet", "sur", "ta", "tandis", "tellement", "tels", "tes", "ton", "tous", "tout", "trop", "très", "tu", "voient", 
        "vont", "votre", "vous", "vu", "ça", "étaient", "état", "étions", "été", "être", "qu'au", "ira", "verrai", "pu", "voir", "j'ai", "«", "»", "toujours", "trs", 
        "jai", "dun", "cest", "mme", "ans", "aprs"
    ]
    
    stop_words_french.update(custom_stopwords)  # Add custom stopwords to the set
    
    # Remove stopwords and punctuation directly by splitting text (no tokenization)
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words_french and word not in string.punctuation])
    
    return text

# Function to remove English stopwords without tokenizing
def remove_english_stopwords(text):
    stop_words_english = set(stopwords.words('english'))  # English stopwords
    
    # Remove stopwords and punctuation directly by splitting text (no tokenization)
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words_english and word not in string.punctuation])
    
    return text