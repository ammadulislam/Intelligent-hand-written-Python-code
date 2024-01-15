import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
loaded_clf = joblib.load(r'C:\Users\HP\PycharmProjects\Fyp_Final\Work\naive_bayes_model.joblib')
loaded_tfidf_vectorizer = joblib.load(r'C:\Users\HP\PycharmProjects\Fyp_Final\Work\tfidf_vectorizer.joblib')

def predict_news_category(text):
    # Preprocess the text using the loaded TF-IDF vectorizer
    text_tfidf = loaded_tfidf_vectorizer.transform([text])

    # Make predictions using the loaded model
    prediction = loaded_clf.predict(text_tfidf)

    # Return the prediction result (0 for fake, 1 for true)
    return prediction[0]

# Example usage
text_to_predict = ("sunday comes after saturday")

prediction_result = predict_news_category(text_to_predict)

if prediction_result == 0:
    print("The given news is predicted to be fake.")
else:
    print("The given news is predicted to be true.")



