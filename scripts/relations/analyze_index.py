import argparse

import pandas as pd

def main(args):
    df = pd.read_csv(args.file_path)

    highest = df.nlargest(5, 'index')
    lowest = df.nsmallest(5, 'index')

    print('Highest')
    for idx, row in highest.iterrows():
        print(row['title'])
    print('Lowest')
    for idx, row in lowest.iterrows():
        print(row['title'])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--file-path')
    args = parser.parse_args()

    main(args)