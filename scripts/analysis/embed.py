import argparse
import os
 
import numpy as np
import pandas as pd

def accumulate_embeddings(df, embeddings_df, feature_cols, len_embed, feature_vals, token_col='token', embedding_col='token_embedding'):
    
    all_embeds_df = None
    for feature_col in feature_cols:
        if feature_col in feature_vals:
            relevant_embeds = embeddings_df[embeddings_df[token_col].isin(feature_vals[feature_col])]
        else:
            relevant_embeds = embeddings_df
        feature_df = df[['id', feature_col]].merge(relevant_embeds, left_on=feature_col, right_on=token_col)
        feature_df = feature_df.drop([feature_col, token_col], axis=1)
        if all_embeds_df is not None:
            all_embeds_df = pd.concat([all_embeds_df, feature_df])
        else:
            all_embeds_df = feature_df

    df = all_embeds_df.groupby('id').apply(np.sum).rename(columns={embedding_col: 'embedding'})
    df = df.reset_index(drop=True)
    return df

def main(args):
    
    embedding_file_name = os.path.basename(args.embeddings)
    embeddings_df = pd.read_csv(args.embeddings)

    # sort out columns
    col_name_mapper = {'Unnamed: 0': 'token'}
    embeddings_df = embeddings_df.rename(columns=col_name_mapper)
    embed_cols = [col for col in embeddings_df.columns.to_list() if col.isdigit()]
    embeddings_df['embedding'] = [np.array(embed) for embed in embeddings_df[embed_cols].values.tolist()]
    embeddings_df = embeddings_df.drop(embed_cols, axis=1)

    article_paths = [os.path.join(args.data_path, file_name) for file_name in os.listdir(args.data_path) if 'news2vec_features' in file_name]

    for article_path in article_paths:

        article_embed_path = article_path.replace('news2vec_features', embedding_file_name)
        if os.path.exists(article_embed_path):
            continue

        print(f"Embedding articles in {os.path.basename(article_path)}")
        article_df = pd.read_csv(article_path)

        col_name_mapper = {'Unnamed: 0': 'id'}
        article_df = article_df.rename(columns=col_name_mapper)

        # creating article embedding
        tfidf_cols = [col for col in article_df.columns.to_list() if col.isdigit()]
        feature_cols = tfidf_cols + ['num_words_ord', 'sentiment', 'month', 'day_of_week', 'day_of_month']
        feature_vals = {
            'num_words_ord': ['wc200', 'wc500', 'wc1000', 'wc2000', 'wc3000', 'wc5000', 'wcmax'],
            'sentiment': ['neutral_1', 'negative_1', 'positive_1'],
            'month': [f"m_{month}" for month in range(1, 13)],
            'day_of_month': [f"d_{day}" for day in range(1, 32)],
            'day_of_week': [f"wd_{day}" for day in range(1, 8)]
        }

        article_embeddings_df = accumulate_embeddings(article_df, embeddings_df, feature_cols, len(embed_cols), feature_vals)

        article_df = article_df[['id', 'title', 'text', 'publish_date', 'url']]
        article_df = article_df.merge(article_embeddings_df, left_on='id', right_on='id')
        article_df = article_df[['title', 'text', 'publish_date', 'url', 'embedding']]

        article_df.to_csv(article_embed_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path')
    parser.add_argument('--embeddings')
    args = parser.parse_args()

    main(args)