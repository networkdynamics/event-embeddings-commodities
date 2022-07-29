import argparse
import datetime
import os
import re

from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import pandas as pd

def main():

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(this_dir_path, '..', '..', 'checkpoints', 'commodity')

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

    methods = [
        'lm_embed',
        'news2vec',
        'sentiment'
    ]

    days_ahead = 30

    for commodity in commodities:
        for method in methods:
            df = pd.read_csv(args.file_path)

            match = re.search("\/([0-9]{1,2})\/", args.file_path)
            days_ahead = int(match.group(0).strip('/'))

            df = df[df['predicted'].notnull()]
            df['date'] = pd.to_datetime(df['date']).dt.date

            df['num_titles'] = df['title'].apply(len)
            #high_attn_df = df[(df['important_title_attention'] > 0.99) & (df['num_titles'] > 80)]

            # plot
            fig, ax = plt.subplots()

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))

            ax.plot(df['date'], df['close'], label='Close')
            ax.plot(df['date'], df['predicted'], label='Predicted Close')

            text_y = 35

            dates = [
                datetime.date(2020, 1, 6),
                datetime.date(2020, 4, 21),
                datetime.date(2020, 11, 4),
                datetime.date(2021, 3, 8)
            ]

            for date in dates:
                date_df = df[df['date'] == date]
                idx = date_df.index.values.astype(int)[0]
                row = date_df.iloc[0]

                ax.text(row['date'], text_y, row['important_title'], {'ha': 'right', 'va': 'top'}, rotation=45, fontsize=13)
                ax.plot([row['date'], row['date']], [text_y, row['close']], 'k-', lw=1, alpha=0.2)
                ax.plot(row['date'], row['close'], 'o', color='blue')

                predicted_row = df.loc[idx + days_ahead]
                future_date = predicted_row['date']
                predicted_val = predicted_row['predicted']
                ax.plot(future_date, predicted_val, 'o', color='orange')


            ax.set_ylim([-50, 75])

    plt.xticks(rotation='45')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()