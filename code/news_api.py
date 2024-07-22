
from transformers import pipeline
from marketaux import Marketaux
import numpy as np

# Sentiment Analysis using BERT
sentiment_pipeline = pipeline('sentiment-analysis')

def get_news_data(api_key, query, from_date, to_date):
    client = Marketaux(api_key)
    news_data = client.get_news(query=query, published_after=from_date, published_before=to_date)
    return news_data['data']

def filter_irrelevant_news(articles, keywords):
    filtered_articles = []
    for article in articles:
        content = (article['title'] + " " + article['description']).lower()
        if any(keyword.lower() in content for keyword in keywords):
            filtered_articles.append(article)
    return filtered_articles

def get_sentiment(text):
    result = sentiment_pipeline(text)
    return result[0]['label'], result[0]['score']

def preprocess_news(articles):
    sentiments = {}
    for article in articles:
        date = article['published_date'][:10]
        text = article['title'] + " " + article['description']
        label, score = get_sentiment(text)
        score = score if label == 'POSITIVE' else -score
        if date in sentiments:
            sentiments[date].append(score)
        else:
            sentiments[date] = [score]
    
    # Average the sentiment scores for each date
    avg_sentiments = {date: np.mean(scores) for date, scores in sentiments.items()}
    return avg_sentiments
