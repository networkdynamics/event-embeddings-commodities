import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

def main(args):

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    relations_path = os.path.join(this_dir_path, '..', '..', 'data', 'relations')
    index_file_path = os.path.join(relations_path, f"{args.country}_threat_regression.csv")
    df = pd.read_csv(index_file_path)

    first_year = 2007
    last_year = 2021

    df = df[['publish_date', 'index']]
    df = df.reset_index()
    df['publish_date'] = pd.to_datetime(df['publish_date'])
    df = df.sort_values('publish_date', ascending=True)
    df = df.groupby(pd.Grouper(key='publish_date', freq='M'))['index'].describe()
    df = df.reset_index()
    #df['moving_average'] = df['mean'].rolling(window=3, center=True, min_periods=3).mean()
    df = df[df['publish_date'].dt.year >= first_year]
    df = df[df['publish_date'].dt.year <= last_year]
    df = df[df['count'] > 10]

    fig, ax1 = plt.subplots()

    ax1.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.yaxis.set_major_locator(mdates.DayLocator(interval=400))

    ln1 = ax1.plot(df['mean'], df['publish_date'])
    
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--country')
    args = parser.parse_args()

    main(args)