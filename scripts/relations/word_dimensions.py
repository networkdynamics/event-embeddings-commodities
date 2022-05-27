import argparse
import os

import numpy as np
import pandas as pd

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir_path, '..', '..', 'data')
    embedding_path = os.path.join(data_path, 'embeddings', '0521_news2vec_embeds.csv')

    df = pd.read_csv(embedding_path)

    col_name_mapper = {'Unnamed: 0': 'token'}
    df = df.rename(columns=col_name_mapper)
    cols = df.columns.to_list()
    cols.remove('token')
    df['embedding'] = [np.array(embed) for embed in df[cols].values.tolist()]
    df = df[['token', 'embedding']]
    
    high_tokens = ['war']
    low_tokens = ['peace']

    high_avg_embed = df[df['token'].isin(high_tokens)][['embedding']].sum().values[0]
    low_avg_embed = df[df['token'].isin(low_tokens)][['embedding']].sum().values[0]
    vec_dim = high_avg_embed - low_avg_embed
    vec_norm = vec_dim / np.linalg.norm(vec_dim)
    vec_origin = (high_avg_embed + low_avg_embed) / 2

    relations_path = os.path.join(data_path, 'relations')
    article_embed_paths = [os.path.join(relations_path, filename) for filename in os.listdir(relations_path) if '0521_news2vec' in filename]

    for article_embed_path in article_embed_paths:
        print(f"Getting index for {os.path.basename(article_embed_path)}")
        df = pd.read_csv(article_embed_path)
        df = df[['publish_date', 'embedding']]
        df['publish_date'] = pd.to_datetime(df['publish_date'])
        df['embedding'] = df['embedding'].str.strip('[]').apply(lambda x: np.fromstring(x, sep=' '))
        df['index'] = df['embedding'].apply(lambda embed: np.dot(embed - vec_origin, vec_norm) / np.linalg.norm(vec_norm))
        df = df[['publish_date', 'index']]
        df = df.groupby('publish_date').describe()
        df.to_csv(article_embed_path.replace('0521_news2vec_embeds', 'war_peace_axis'))


if __name__ == '__main__':
    main()