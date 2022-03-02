import os

import pandas as pd
import matplotlib.pyplot as plt

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir_path, '..', '..', 'data')
    csv_path = os.path.join(data_path, 'articles_in_months.csv')

    df = pd.read_csv(csv_path)

    df['date'] = df['month'].map(str)+ '-' +df['year'].map(str)
    df['date'] = pd.to_datetime(df['date'], format='%m-%Y').dt.strftime('%m-%Y')
    plt.plot_date(df['date'], df['Value'])
    plt.show()

if __name__ == '__main__':
    main()