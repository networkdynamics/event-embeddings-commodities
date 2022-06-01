import argparse

from matplotlib import pyplot as plt
import pandas as pd

def main(args):

    df = pd.read_csv(args.file_path)

    df = df[['publish_date', 'index']]
    df = df.reset_index()
    df['publish_date'] = pd.to_datetime(df['publish_date'])
    df = df.sort_values('publish_date', ascending=True)
    df = df.groupby(pd.Grouper(key='publish_date', freq='M'))['index'].describe()
    df = df.reset_index()
    plt.plot(df['publish_date'], df['mean'])
    plt.plot(df['publish_date'], df['25%'])
    plt.plot(df['publish_date'], df['75%'])
    plt.plot(df['publish_date'], df['min'])
    plt.plot(df['publish_date'], df['max'])
    plt.xticks(rotation='vertical')
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--file-path')
    args = parser.parse_args()

    main(args)