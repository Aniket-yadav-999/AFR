from flask import Flask, render_template, request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load sentiment models
vader_analyzer = SentimentIntensityAnalyzer()
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
roberta_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# VADER analysis
def get_vader_scores(text):
    scores = vader_analyzer.polarity_scores(text)
    return {
        'sentiments': ['Negative 😠', 'Neutral 😐', 'Positive 😊'],
        'scores': [scores['neg'], scores['neu'], scores['pos']]
    }

# RoBERTa analysis
def get_roberta_scores(text):
    encoded = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        output = roberta_model(**encoded)
    scores = softmax(output.logits[0].numpy())
    return {
        'sentiments': ['Negative 😠', 'Neutral 😐', 'Positive 😊'],
        'scores': [scores[0], scores[1], scores[2]]
    }

# Interpretation logic
def interpret(vader, roberta):
    def max_sentiment(scores):
        idx = np.argmax(scores)
        return ['Negative', 'Neutral', 'Positive'][idx], scores[idx]

    vader_label, vader_conf = max_sentiment(vader['scores'])
    roberta_label, roberta_conf = max_sentiment(roberta['scores'])

    if vader_label == roberta_label:
        return f"✅ Both models predict the sentiment as <strong>{vader_label}</strong>.<br>Confidence: VADER - {vader_conf*100:.1f}%, RoBERTa - {roberta_conf*100:.1f}%"
    else:
        return f"⚠️ VADER says <strong>{vader_label}</strong> ({vader_conf*100:.1f}%)<br>🤗 RoBERTa says <strong>{roberta_label}</strong> ({roberta_conf*100:.1f}%)"

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    review = ""
    vader_result = None
    roberta_result = None
    interpretation = ""
    vader_zipped = []
    roberta_zipped = []

    if request.method == 'POST':
        review = request.form['review']
        if review.strip():
            try:
                vader_result = get_vader_scores(review)
                roberta_result = get_roberta_scores(review)

                # Zip sentiment and score lists
                vader_zipped = list(zip(vader_result['sentiments'], vader_result['scores']))
                roberta_zipped = list(zip(roberta_result['sentiments'], roberta_result['scores']))

                interpretation = interpret(vader_result, roberta_result)

            except Exception as e:
                interpretation = f"<span style='color: red;'>Error: {e}</span>"

    return render_template("index.html",
                           review=review,
                           vader_zipped=vader_zipped,
                           roberta_zipped=roberta_zipped,
                           interpretation=interpretation)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
