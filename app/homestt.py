import streamlit as st

# Title and Subheader
st.title("Text Review Prediction App")
st.subheader("Welcome to the Text Review Prediction App!")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page:", ["Home", "Model Prediction", "LLM Prediction"])

# Display content depending on the selected page
if page == "Home":
    st.write("Welcome to the Home page! Select a model for prediction.")
elif page == "Model Prediction":
    # Redirect to the model prediction page (model.py)
    st.write("You selected Model Prediction.")
    import main  # Import the model prediction logic from model.py
    main.run()   # Run the model prediction code
elif page == "LLM Prediction":
    # Redirect to the LLM prediction page (llm.py)
    st.write("You selected LLM Prediction. ")
    import llmst  # Import the LLM prediction logic from llm.py
    llmst.run()   # Run the LLM prediction code
