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

    #plot
    num_rows = 2
    num_cols = 7
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(16,10))

    for idx, commodity in enumerate(commodities):
        row_idx = idx // num_cols
        col_idx = idx % num_cols
        ax = axes[row_idx, col_idx]
        plotted_truth = False

        commodity_dir_path = os.path.join(checkpoint_path, commodity)
        for method in methods:
            method_dir_path = os.path.join(commodity_dir_path, str(days_ahead), method)
            prediction_file_names = os.listdir(method_dir_path)
            best_accuracy = 999999
            for file_name in prediction_file_names:
                if not file_name.endswith('predictions.csv'):
                    continue

                if 'dir' in file_name or 'diff' in file_name:
                    continue

                acc_regex = '[0-9]{1}_[0-9]{4}'
                acc_match = re.search(acc_regex, file_name)
                if not acc_match:
                    continue

                acc_str = acc_match.group(0)
                accuracy = float(acc_str.replace('_', '.'))
                if accuracy < best_accuracy:
                    best_accuracy = accuracy
                    best_predictions_name = file_name

            predictions_path = os.path.join(method_dir_path, best_predictions_name)
            df = pd.read_csv(predictions_path)

            df = df[df['predicted'].notnull()]
            df['date'] = pd.to_datetime(df['date']).dt.date

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))

            if not plotted_truth:
                ax.plot(df['date'], df['close'], label='Close')
                plotted_truth = True

            ax.plot(df['date'], df['predicted'], label=f'{method} Predicted')
            ax.set_title(commodity)

    plt.legend()
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()