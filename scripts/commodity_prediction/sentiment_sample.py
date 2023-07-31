import os

import pandas as pd

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', '..', 'data', 'commodity_data')

    commodities = [
        'crude_oil',
        'brent_crude_oil',
        'natural_gas',
        'rbob_gasoline',
        'copper',
        'palladium',
        'platinum',
        'gold',
        'silver',
        'corn',
        'cotton',
        'soybean',
        'sugar',
        'wheat'
    ]

    for commodity in commodities:
        commodity_sentiment_path = os.path.join(data_dir_path, f'{commodity}_sentiment.csv')

        comm_sent_df = pd.read_csv(commodity_sentiment_path)
        comm_sent_df = comm_sent_df[['title', 'text', 'sentiment']]
        examples = comm_sent_df.to_dict('records')
        print(comm_sent_df.head())

if __name__ == '__main__':
    main()