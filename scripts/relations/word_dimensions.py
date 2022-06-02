import argparse
import os

import numpy as np
import pandas as pd
import torch
import transformers

def main(args):
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir_path, '..', '..', 'data')
    relations_path = os.path.join(data_path, 'relations')

    if args.label_method == 'words':
        high_tokens = ['invasion', 'armies', 'terrorism', 'fascism', 'aggressors', 'attack', 'troops', 'spies']
        low_tokens = ['diplomacy', 'peacemaking', 'mediate', 'cooperative', 'embassies']

        if args.embed_method == 'news2vec':
            embedding_path = os.path.join(data_path, 'embeddings', '0521_news2vec_embeds.csv')

            df = pd.read_csv(embedding_path)

            col_name_mapper = {'Unnamed: 0': 'token'}
            df = df.rename(columns=col_name_mapper)
            cols = df.columns.to_list()
            cols.remove('token')
            df['embedding'] = [np.array(embed) for embed in df[cols].values.tolist()]
            df = df[['token', 'embedding']]
            
            high_avg_embed = df[df['token'].isin(high_tokens)]['embedding'].mean()
            low_avg_embed = df[df['token'].isin(low_tokens)]['embedding'].mean()

            path_keyword = '0521_news2vec_embed'

        elif args.embed_method == 'lm_embed':
            tokenizer = transformers.RobertaTokenizerFast.from_pretrained('roberta-base')
            lm_model = transformers.RobertaModel.from_pretrained('roberta-base')

            embed_size = 768
            high_embeddings = np.empty((len(high_tokens), embed_size))
            for idx, word in enumerate(high_tokens):
                inputs = tokenizer(word, return_tensors="pt")
                outputs = lm_model(**inputs)
                high_embeddings[idx, :] = outputs.last_hidden_state[:, 0, :].detach().numpy()
            high_avg_embed = np.mean(high_embeddings, axis=0)

            low_embeddings = np.empty((len(low_tokens), embed_size))
            for idx, word in enumerate(low_tokens):
                inputs = tokenizer(word, return_tensors="pt")
                outputs = lm_model(**inputs)
                low_embeddings[idx, :] = outputs.last_hidden_state[:, 0, :].detach().numpy()
            low_avg_embed = np.mean(low_embeddings, axis=0)

            path_keyword = 'lm_embed'

    elif args.label_method == 'articles':
        if not args.embed_method == 'lm_embed':
            raise Exception()

        path_keyword = 'lm_embed'
        high_titles = [
            'Russia deploys more surface-to-air missiles in Crimean build-up', 
            "Canadian PM says mosque shooting a 'terrorist attack on Muslims'", 
            "Clinton: Iran moving toward military dictatorship",
            "Two rockets fall inside Iraqi air base housing U.S. troops: security sources",
            "U.S. accuses Russian spies of 2016 election hacking as summit looms"
            ]
        low_titles = [
            'Britain says in talks with Iran about reopening embassies', 
            "Africa's emerging middle class drives growth and democracy", 
            "Bush to visit biblical site on peacemaking tour", 
            '''"At U.N., Congo's Kabila vows 'peaceful, credible' elections"''', 
            '''"U.S., Japan to cooperate on energy, infrastructure investment: Treasury"''']

        us_embed_path = os.path.join(relations_path, 'united_states_lm_embed.csv')
        df = pd.read_csv(us_embed_path)
        df = df.rename(columns={'embed': 'embedding'})
        df = df[['publish_date', 'title', 'embedding']]
        df['publish_date'] = pd.to_datetime(df['publish_date'])
        df['embedding'] = df['embedding'].str.strip('[]').apply(lambda x: np.fromstring(x, sep=' '))

        high_embeds = df[df['title'].isin(high_titles)]
        low_embeds = df[df['title'].isin(low_titles)]
        high_avg_embed = high_embeds['embedding'].mean()
        low_avg_embed = low_embeds['embedding'].mean()


    vec_dim = high_avg_embed - low_avg_embed
    vec_norm = vec_dim / np.linalg.norm(vec_dim)
    vec_origin = (high_avg_embed + low_avg_embed) / 2

    article_embed_paths = [os.path.join(relations_path, filename) for filename in os.listdir(relations_path) if path_keyword in filename]

    for article_embed_path in article_embed_paths:
        print(f"Getting index for {os.path.basename(article_embed_path)}")
        df = pd.read_csv(article_embed_path)
        df = df.rename(columns={'embed': 'embedding'})
        df = df[['publish_date', 'title', 'embedding']]
        df['publish_date'] = pd.to_datetime(df['publish_date'])
        df['embedding'] = df['embedding'].str.strip('[]').apply(lambda x: np.fromstring(x, sep=' '))
        df['index'] = df['embedding'].apply(lambda embed: np.dot(embed - vec_origin, vec_norm) / np.linalg.norm(vec_norm))
        df = df[['publish_date', 'title', 'index']]
        df.to_csv(article_embed_path.replace(path_keyword, 'war_peace_axis'), index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--embed-method')
    parser.add_argument('--label-method')
    args = parser.parse_args()

    main(args)