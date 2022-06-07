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

    fed_index_path = os.path.join(relations_path, 'data_gpr_export.xls')
    fed_index_df = pd.read_excel(fed_index_path)

    country_map = {
        'united_states': 'USA'
    }
    fed_index_df = fed_index_df[['month', f'GPRC_{country_map[args.country]}']]
    fed_index_df = fed_index_df[fed_index_df['month'].dt.year >= first_year]
    fed_index_df = fed_index_df[fed_index_df['month'].dt.year <= last_year]

    fig, ax1 = plt.subplots()

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=400))

    ln1 = ax1.plot(df['publish_date'], df['mean'], label='Embed Index', color='r')

    ax2 = ax1.twinx()

    ln2 = ax2.plot(fed_index_df['month'], fed_index_df['GPRC_USA'], label='Fed Index')

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    
    fig.tight_layout()
    plt.xticks(rotation='vertical')
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--country')
    args = parser.parse_args()

    main(args)