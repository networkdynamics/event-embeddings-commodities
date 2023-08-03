import os
import re

import nltk
import pandas as pd
from tqdm import tqdm
import transformers

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
    'cotton': ['cotton'],
    'soybean': ['soybean', 'soybeans'],
    'sugar': ['sugar'],
    'wheat': ['wheat']
}

class CommoditySentiment:
    def __init__(self, commodity):
        from NewsSentiment import TargetSentimentClassifier
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
            return None


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

class FinancialSentiment:
    def __init__(self):

        finbert = transformers.BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
        tokenizer = transformers.BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

        self.classifier = transformers.pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer, device=0, return_all_scores=True)

    def get_sentiment(self, txt):
        tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}

        results = self.classifier(txt, **tokenizer_kwargs)
        prob_positive = [sentiment_class for sentiment_class in results[0] if sentiment_class['label'] == 'Positive'][0]['score']
        return prob_positive


def main():

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir_path, '..', '..', 'data')

    if not os.path.exists(data_path):
        os.mkdir(data_path)

    tqdm.pandas()

    classifier_type = 'financial'

    for commodity in COMMODITIES:
        if classifier_type == 'targeted':
            classifier = CommoditySentiment(commodity)
            suffix = 'sentiment'
        elif classifier_type == 'financial':
            classifier = FinancialSentiment()
            suffix = 'fsentiment'

        csv_path = os.path.join(data_path, 'commodity_data', f"{commodity}_articles.csv")
        out_path = os.path.join(data_path, 'commodity_data', f"{commodity}_{suffix}.csv")

        if os.path.exists(out_path):
            continue

        articles_df = pd.read_csv(csv_path)

        articles_df['all_text'] = articles_df[['title', 'text']].agg('. '.join, axis=1)
        articles_df['sentiment'] = articles_df['all_text'].progress_apply(classifier.get_sentiment)
        articles_df = articles_df[articles_df['sentiment'].notnull()]
        articles_df = articles_df.drop(columns=['all_text'])

        articles_df.to_csv(out_path)


if __name__ == '__main__':
    main()