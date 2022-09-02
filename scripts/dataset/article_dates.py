import datetime
import os

import pandas as pd
import matplotlib.pyplot as plt

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir_path, '..', '..', 'data')
    csv_path = os.path.join(data_path, 'articles_in_months.csv')

    df = pd.read_csv(csv_path)

    df = df[~df['date_month'].isna()]
    df = df[['date_year', 'count']]
    df = df.groupby('date_year').sum('count')
    df = df.reset_index()
    df['date'] = df['date_year'].map(lambda x: str(int(x)))
    df['date'] = pd.to_datetime(df['date'], format='%Y')
    plt.plot(df['date'], df['count'])
    # plt.show()
    # df['date'] = df['date_month'].map(lambda x: str(int(x)))+ '-' +df['date_year'].map(lambda x: str(int(x)))
    # df['date'] = pd.to_datetime(df['date'], format='%m-%Y')
    # df = df.sort_values(by='date')
    # plt.plot(df['date'], df['count'])

    # find 1 million under curve
    start_year = 2007
    start_date = datetime.date(start_year, 1, 1)
    #end_date = datetime.date(2021, 10, 31)
    end_date = datetime.date(2021, 1, 1)

    #num_months = ((end_date.year - start_date.year) * 12) + (end_date.month - start_date.month) + 1
    num_years = end_date.year - start_date.year + 1
    #months = [datetime.date(start_year + (i // 12), 1 + (i % 12), 1) for i in range(num_months)]
    years = [datetime.date(start_year + i, 1, 1) for i in range(num_years)]
    #counts = [1000000 // num_months for _ in range(num_months)]

    #counts = [145 * i for i in range(20)] + [2900 for _ in range(50)] + [2900 + (i * 100) for i in range(30)] + [5900 for _ in range(5)] + [5900 + (i * 250) for i in range(20)] + [11800 for _ in range(40)] + [400 * (13 - i) for i in range(13)]
    counts = [18000, 40000, 42000, 44000, 44000, 54000, 57000, 78000, 93000, 100000, 100000, 100000, 100000, 100000, 45000]

    print(sum(counts))
    #plt.plot(months, counts)
    #plt.plot(years, counts)

    plt.show()

if __name__ == '__main__':
    main()