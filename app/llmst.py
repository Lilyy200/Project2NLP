import streamlit as st
import requests

# Configuration de l'API
BASE_URL = "http://192.168.1.57:1234"
MODEL_NAME = "llama-3.2-3b-instruct"
ENDPOINT = f"{BASE_URL}/v1/chat/completions"

def get_llm_response(user_input):
    """
    Fonction pour interroger le modèle LLM via une API REST.

    Args:
        user_input (str): Le texte de l'utilisateur à envoyer au modèle.

    Returns:
        str: La réponse du modèle, ou un message d'erreur en cas d'échec.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": f"Given the following review for an insurance company: {user_input} predict a star rating going from 1-5: "}
        ],
        "max_tokens": 128,
        "temperature": 0.7,
        "top_p": 1.0,
    }

    try:
        response = requests.post(ENDPOINT, json=payload)
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

def run():
    """Fonction principale pour la page Streamlit de LLM."""
    # Titre et description de la page
    st.title("LLM Prediction")
    st.subheader("Use the LLM to predict a star rating for an insurance review")

    # Zone de saisie pour l'utilisateur
    user_input = st.text_area("Enter your insurance review:", height=200)

    # Bouton pour déclencher la prédiction
    if st.button("Predict"):
        if user_input:
            # Appel à la fonction pour obtenir la réponse du LLM
            with st.spinner("Querying the model..."):
                response = get_llm_response(user_input)
            
            # Affichage de la réponse du modèle
            st.write("**Model Response:**")
            st.success(response)
        else:
            st.error("Please enter a review before clicking Predict.")

# Exécute la fonction principale si le fichier est appelé directement
if __name__ == "__main__":
    run()
