
import pandas as pd
from textblob import TextBlob

def compute_sentiment(df, text_col='review_text'):
    df = df.copy()
    df[text_col] = df[text_col].fillna("").astype(str)
    def polarity(text):
        try:
            return TextBlob(text).sentiment.polarity
        except:
            return 0.0
    df['polarity'] = df[text_col].apply(polarity)
    df['sentiment'] = df['polarity'].apply(lambda x: 'positive' if x>0.1 else ('negative' if x<-0.1 else 'neutral'))
    return df
