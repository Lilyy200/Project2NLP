# Dictionary to map model names to file paths
model_files = {
    "Linear Regression": "models/Linear Regression_model.pkl",
    "SVM": "models/SVM_model.pkl",
    "Random Forest": "models/Random Forest_model.pkl",
    "AdaBoost": "models/AdaBoost_model.pkl",
    "Gradient Boosting": "models/Gradient Boosting_model.pkl",
    "pretrained embedding": "models/my_pretrained_embedding_model.h5",  # Assuming BERT is a Keras model
    "use": "models/use_model.pkl",  # Assuming USE is a scikit-learn model
    "my_embedding": "models/my_embedding_model.h5",  # Keras model for your custom embedding
    "sentence_bert": "models/sentence_bert_model.pkl",  # Assuming sentence BERT is a scikit-learn model
    "word2vec": "models/word2vec_model.model",  # Assuming Word2Vec is stored with its .model extension
    "xgb_regressor": "models/xgb_regressor_model.pkl"  # XGBoost model
}
