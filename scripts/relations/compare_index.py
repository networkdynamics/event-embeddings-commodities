import argparse
import datetime
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import pyreadr

def main():

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    relations_path = os.path.join(this_dir_path, '..', '..', 'data', 'relations')
    country = 'united_states'
    embed = 'lm_small_embed'
    method = 'logistic'
    index_file_path = os.path.join(relations_path, f"{country}_{embed}_threat_{method}.csv")
    df = pd.read_csv(index_file_path)

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
    df['mean'] = (df['mean'] - df['mean'].mean()) / df['mean'].std()

    # load fed index
    fed_index_path = os.path.join(relations_path, 'fedreserve', 'data_gpr_export.xls')
    fed_index_df = pd.read_excel(fed_index_path)

    fed_index_df = fed_index_df[['month', 'GPRC_USA']]
    fed_index_df = fed_index_df[fed_index_df['month'].dt.year >= first_year]
    fed_index_df = fed_index_df[fed_index_df['month'].dt.year <= last_year]
    fed_index_df['GPRC_USA'] = (fed_index_df['GPRC_USA'] - fed_index_df['GPRC_USA'].mean()) / fed_index_df['GPRC_USA'].std()

    # load trubowitz index
    trubowitz_index_path = os.path.join(relations_path, 'trubowitzpaper', 'data_gpr.RDS')
    trubowitz_index_df = pyreadr.read_r(trubowitz_index_path)[None]
    trubowitz_index_df = trubowitz_index_df[['Date', 'GPR', 'GPR_ACT']]
    trubowitz_index_df['Date'] = pd.to_datetime(trubowitz_index_df['Date'], format='%Y-%d-%m')
    trubowitz_index_df = trubowitz_index_df[trubowitz_index_df['Date'].dt.year >= first_year]
    trubowitz_index_df = trubowitz_index_df[trubowitz_index_df['Date'].dt.year <= last_year]
    trubowitz_index_df['GPR'] = (trubowitz_index_df['GPR'] - trubowitz_index_df['GPR'].mean()) / trubowitz_index_df['GPR'].std()
    trubowitz_index_df['GPR_ACT'] = (trubowitz_index_df['GPR_ACT'] - trubowitz_index_df['GPR_ACT'].mean()) / trubowitz_index_df['GPR_ACT'].std()
    
    fig, ax = plt.subplots(figsize=(12,9))

    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    #ax.xaxis.set_major_locator(mdates.DayLocator(interval=400))

    ax.plot(df['publish_date'], df['mean'], label='Embed Index', color='r')
    ax.plot(fed_index_df['month'], fed_index_df['GPRC_USA'], label='Fed Index')
    ax.plot(trubowitz_index_df['Date'], trubowitz_index_df['GPR'], label='Trubowitz GPR')
    ax.plot(trubowitz_index_df['Date'], trubowitz_index_df['GPR_ACT'], label='Trubowitz GPR ACT')

    ax.legend(loc='upper left')

    # add text
    props = {'ha': 'left', 'va': 'bottom'}
    lower_y_lim = -3

    events = [
        (datetime.date(2010, 2, 1), 'Peak of War in Afghanistan'),
        (datetime.date(2011, 3, 1), 'Libya Intervention'),
        (datetime.date(2014, 2, 1), 'Russia annexes Crimea'),
        (datetime.date(2015, 11, 1), 'Paris attacks'),
        (datetime.date(2017, 9, 1), 'US - North Korea Tensions'),
        (datetime.date(2018, 4, 1), 'Syria missile strikes'),
        (datetime.date(2018, 7, 1), 'US - Iran Tensions'),
        (datetime.date(2019, 9, 1), 'Attack on Saudi Oil'),
        (datetime.date(2020, 1, 1), 'Iranian General Killed')
    ]
    for event in events:
        ax.plot([event[0], event[0]], [lower_y_lim, 4], 'k-', lw=1, alpha=0.2)
        ax.text(event[0], 4, event[1], props, rotation=45)
    
    ax.set_ylim([lower_y_lim, 8.5])

    plt.xticks(rotation='45')

    fig_path = os.path.join(relations_path, '..', '..', 'figs', f'{country}_threat_compare_{embed}_{method}.png')
    plt.savefig(fig_path)


if __name__ == '__main__':
    main()