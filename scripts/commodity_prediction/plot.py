import argparse
import os

from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import pandas as pd

def main(args):

    df = pd.read_csv(args.file_path)

    df = df[df['predicted'].notnull()]
    df['date'] = pd.to_datetime(df['date'])

    # plot
    fig, ax = plt.subplots()

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=20))

    ax.plot(df['date'], df['close'], label='Close')
    ax.plot(df['date'], df['close'].shift(30), label='Continues')
    ax.plot(df['date'], df['predicted'], label='Predicted Close')

    plt.xticks(rotation='45')
    plt.legend()
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--file-path')
    args = parser.parse_args()

    main(args)