import argparse
import datetime
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

def main(args):

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    relations_path = os.path.join(this_dir_path, '..', '..', 'data', 'relations')
    index_file_path = os.path.join(relations_path, f"{args.country}_threat_regression.csv")
    df = pd.read_csv(args.file_path)

    first_year = 2007
    last_year = 2021

    df = df[['publish_date', 'index']]
    df = df.reset_index()
    df['publish_date'] = pd.to_datetime(df['publish_date'])
    df = df.sort_values('publish_date', ascending=True)
    df = df.groupby(pd.Grouper(key='publish_date', freq='MS'))['index'].describe()
    df = df.reset_index()
    #df['moving_average'] = df['mean'].rolling(window=3, center=True, min_periods=3).mean()
    df = df[df['publish_date'].dt.year >= first_year]
    df = df[df['publish_date'].dt.year <= last_year]
    df = df[df['count'] > 10]

    fed_index_path = os.path.join(relations_path, 'data_gpr_export.xls')
    fed_index_df = pd.read_excel(fed_index_path)

    country_map = {
        'united_states': 'USA'
    }
    fed_index_df = fed_index_df[['month', f'GPRC_{country_map[args.country]}']]
    fed_index_df = fed_index_df[fed_index_df['month'].dt.year >= first_year]
    fed_index_df = fed_index_df[fed_index_df['month'].dt.year <= last_year]

    df['mean'] = (df['mean'] - df['mean'].mean()) / df['mean'].std()
    fed_index_df['GPRC_USA'] = (fed_index_df['GPRC_USA'] - fed_index_df['GPRC_USA'].mean()) / fed_index_df['GPRC_USA'].std()

    fig, ax = plt.subplots()

    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    #ax.xaxis.set_major_locator(mdates.DayLocator(interval=400))

    ax.plot(df['publish_date'], df['mean'], label='Embed Index', color='r')
    ax.plot(fed_index_df['month'], fed_index_df['GPRC_USA'], label='Fed Index')

    ax.legend(loc='upper left')

    # add text
    props = {'ha': 'left', 'va': 'bottom'}

    events = [
        (datetime.date(2011, 3, 1), 'Libya Intervention'),
        (datetime.date(2014, 2, 1), 'Russia annexes Crimea'),
        (datetime.date(2017, 9, 1), 'US - North Korea Tensions')
    ]
    for event in events:
        ax.plot([event[0], event[0]], [0, 4], 'k-', lw=1, alpha=0.2)
        ax.text(event[0], 4, event[1], props, rotation=45)
    
    ax.set_ylim([-3, 8])

    plt.xticks(rotation='45')
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--country')
    parser.add_argument('--file-path')
    args = parser.parse_args()

    main(args)