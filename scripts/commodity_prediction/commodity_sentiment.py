import os
import re

import nltk
from NewsSentiment import TargetSentimentClassifier
import pandas as pd

COMMODITIES = {
    'brent_crude_oil': ['oil', 'crude', 'brent'],
    'crude_oil': ['oil', 'crude'],
    'natural_gas': ['gas'],
    'rbob_gasoline': ['gasoline', 'rbob'],
    'copper': ['copper'],
    'palladium': ['palladium'],
    'platinum': ['platinum'],
    'gold': ['gold'],
    'silver': ['silver'],
    'corn': ['corn'],
    'oat': ['oat', 'oats'],
    'cotton': ['cotton'],
    'lumber': ['lumber'],
    'cocoa': ['cocoa'],
    'coffee': ['coffee'],
    'feeder_cattle': ['cattle'],
    'heating_oil': ['oil', 'heating'],
    'lean_hogs': ['hogs'],
    'live_cattle': ['cattle'],
    'soybean_meal': ['soybean', 'soybeans'],
    'soybean_oil': ['soybean', 'soybeans'],
    'soybean': ['soybean', 'soybeans'],
    'sugar': ['sugar'],
    'wheat': ['wheat']
}

class CommoditySentiment:
    def __init__(self, commodity):
        self.sentiment_classifier = TargetSentimentClassifier()
        self.commodity = commodity

    def get_sentiment(self, txt):

        txt = txt.replace('\n', ' ').lower()

        keywords = COMMODITIES[self.commodity]
        for keyword in keywords:
            phrase = r'\b' + keyword + r'\b'
            match = re.search(phrase, txt)
            if match:
                break
        else:
            print(f"Keywords: {keywords} not found in text: {txt}")

        left = txt[:match.start()]
        centre = txt[match.start():match.end()]
        right = txt[match.end():]

        try:
            sentiment = self.sentiment_classifier.infer_from_text(left, centre, right)
        except Exception:
            sentences = nltk.tokenize.sent_tokenize(txt)
            
            for sentence in sentences:
                match = re.search(phrase, sentence)
                if match:
                    break

            left = sentence[:match.start()]
            centre = sentence[match.start():match.end()]
            right = sentence[match.end():]
            try:
                sentiment = self.sentiment_classifier.infer_from_text(left, centre, right)
            except Exception as err:
                return None

        prob_positive = [sentiment_class for sentiment_class in sentiment if sentiment_class['class_label'] == 'positive'][0]['class_prob']
        return prob_positive

def main():

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir_path, '..', '..', 'data')

    if not os.path.exists(data_path):
        os.mkdir(data_path)

    for commodity in COMMODITIES:
        csv_path = os.path.join(data_path, 'commodity_data', f"{commodity}_articles.csv")
        
        classifier = CommoditySentiment(commodity)

        articles_df = pd.read_csv(csv_path)

        articles_df['all_text'] = articles_df[['title', 'text']].agg('. '.join, axis=1)
        articles_df['sentiment'] = articles_df['all_text'].apply(classifier.get_sentiment)
        articles_df = articles_df[articles_df['sentiment'].notnull()]

        out_path = os.path.join(data_path, 'commodity_data', f"{commodity}_sentiment.csv")
        articles_df.to_csv(out_path)


if __name__ == '__main__':
    main()