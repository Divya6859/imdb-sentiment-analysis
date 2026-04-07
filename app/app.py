import gradio as gr
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load saved model & vectorizer
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Prediction function
def predict_sentiment(review):
    cleaned = clean_text(review)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]
    
    if prediction == 1:
        return "Positive 😊"
    else:
        return "Negative 😞"

# Gradio Interface
interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=5, placeholder="Enter movie review here..."),
    outputs="text",
    title="🎬 IMDb Movie Review Sentiment Analysis",
    description="Enter a movie review and get sentiment prediction (Positive/Negative)"
)

interface.launch(share=True)
