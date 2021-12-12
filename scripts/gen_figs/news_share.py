import os

from matplotlib import pyplot as plt
import pandas as pd

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    news_share_path = os.path.join(this_dir_path, '..', '..', 'data', 'news_share', 'cc_news_share_data.csv')
    df = pd.read_csv(news_share_path)
    df = df.pivot(index='crawl', columns='url_host_registered_domain', values='count')
    df = df.sort_index()

    df.plot()
    plt.show()


if __name__ == '__main__':
    main()
