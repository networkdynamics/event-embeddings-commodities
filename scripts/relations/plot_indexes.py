import argparse
import datetime
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

def main():

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    relations_path = os.path.join(this_dir_path, '..', '..', 'data', 'relations')

    fig, axes = plt.subplots(nrows=2, ncols=3)

    first_year = 2007
    last_year = 2021

    fed_index_path = os.path.join(relations_path, 'data_gpr_export.xls')
    fed_index_df = pd.read_excel(fed_index_path)

    country_map = {
        'united_states': 'USA',
        'united_kingdom': 'GBR',
        'china': 'CHN',
        'germany': 'DEU',
        'france': 'FRA',
        'russia': 'RUS'
    }

    countries = [
        'united_states',
        'united_kingdom',
        'france',
        'germany',
        'china',
        'russia'
    ]

    for row_idx, row in enumerate(axes):
        for col_idx, ax in enumerate(row):
            country = countries[(row_idx * len(row)) + col_idx]

            index_file_path = os.path.join(relations_path, f"{country}_threat_regression.csv")
            df = pd.read_csv(index_file_path)

            df = df[['publish_date', 'index']]
            df = df.reset_index()
            df['publish_date'] = pd.to_datetime(df['publish_date'])
            df = df.sort_values('publish_date', ascending=True)
            df = df.groupby(pd.Grouper(key='publish_date', freq='MS'))['index'].describe()
            df = df.reset_index()
            
            df = df[df['publish_date'].dt.year >= first_year]
            df = df[df['publish_date'].dt.year <= last_year]
            df = df[df['count'] > 10]

            country_col_name = f'GPRC_{country_map[country]}'
            fed_index_df = fed_index_df[fed_index_df['month'].dt.year >= first_year]
            fed_index_df = fed_index_df[fed_index_df['month'].dt.year <= last_year]

            df['mean'] = (df['mean'] - df['mean'].mean()) / df['mean'].std()
            fed_index = (fed_index_df[country_col_name] - fed_index_df[country_col_name].mean()) / fed_index_df[country_col_name].std()

            window_size = 50
            df_moving_avg = df['mean'].rolling(window=window_size, center=True, min_periods=1).mean()
            fed_moving_avg = fed_index.rolling(window=window_size, center=True, min_periods=1).mean()

            #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            #ax.xaxis.set_major_locator(mdates.DayLocator(interval=400))

            human_name = ' '.join([word[0].upper() + word[1:] for word in country.split('_')])
            ax.set_title(human_name)

            ax.plot(df['publish_date'], df_moving_avg, label='Embed Index', color='r')
            ax.plot(fed_index_df['month'], fed_moving_avg, label='Fed Index')

            ax.legend(loc='upper left')

    plt.show()


if __name__ == '__main__':

    main()