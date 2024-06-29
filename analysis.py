from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer


import nltk
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Perform sentiment analysis on a chapter
def analyze_sentiment(chapter):
    
    sentiment = sia.polarity_scores(chapter)
    # blob = TextBlob(chapter)
    # sentiment = blob.sentiment.polarity
    return sentiment