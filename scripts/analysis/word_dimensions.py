import os

import numpy as np
import pandas as pd

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir_path, '..', '..', 'data')
    embedding_path = os.path.join(data_path, '0713_21_articles.emb')

    df = pd.read_csv(embedding_path)

    col_name_mapper = {'Unnamed: 0': 'token'}
    df = df.rename(columns=col_name_mapper)
    cols = df.columns.to_list()
    cols.remove('token')
    df['embedding'] = [np.array(embed) for embed in df[cols].values.tolist()]
    df = df[['token', 'embedding']]
    
    high_tokens = ['war']
    low_tokens = ['peace']

    high_avg_embed = df[df['token'].isin(high_tokens)][['embedding']].sum()
    low_avg_embed = df[df['token'].isin(low_tokens)][['embedding']].sum()
    vec_dim = high_avg_embed - low_avg_embed
    vec_norm = vec_dim / np.linalg.norm(vec_dim)
    vec_origin = (high_avg_embed + low_avg_embed) / 2

    df = pd.read_csv(os.path.join(data_path, 'article_embeds.csv'))
    df['index'] = df['embedding'].apply(lambda embed: np.dot(embed - vec_origin, vec_norm) / np.linalg.norm(vec_norm))


if __name__ == '__main__':
    main()