import argparse
import os

import numpy as np
import pandas as pd
import sklearn
import transformers
from sklearn.linear_model import LinearRegression

def main(args):
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir_path, '..', '..', 'data')
    relations_path = os.path.join(data_path, 'relations')

    high_tokens = ['invasion', 'armies', 'terrorism', 'fascism', 'aggressors', 'attack', 'troops', 'spies']
    low_tokens = ['diplomacy', 'peacemaking', 'mediate', 'cooperative', 'embassies']
    neutral_tokens = ['stock', 'index']

    tokenizer = transformers.RobertaTokenizerFast.from_pretrained('roberta-base')
    lm_model = transformers.RobertaModel.from_pretrained('roberta-base')

    embed_size = 768
    high_embeddings = np.empty((len(high_tokens), embed_size))
    for idx, word in enumerate(high_tokens):
        inputs = tokenizer(word, return_tensors="pt")
        outputs = lm_model(**inputs)
        high_embeddings[idx, :] = outputs.last_hidden_state[:, 0, :].detach().numpy()

    low_embeddings = np.empty((len(low_tokens), embed_size))
    for idx, word in enumerate(low_tokens):
        inputs = tokenizer(word, return_tensors="pt")
        outputs = lm_model(**inputs)
        low_embeddings[idx, :] = outputs.last_hidden_state[:, 0, :].detach().numpy()

    neutral_embeddings = np.empty((len(neutral_tokens), embed_size))
    for idx, word in enumerate(neutral_tokens):
        inputs = tokenizer(word, return_tensors="pt")
        outputs = lm_model(**inputs)
        neutral_embeddings[idx, :] = outputs.last_hidden_state[:, 0, :].detach().numpy()

    path_keyword = 'lm_embed'

    if args.model_method == 'avg':
        high_avg_embed = np.mean(high_embeddings)
        low_avg_embed = np.mean(low_embeddings)
        vec_dim = high_avg_embed - low_avg_embed
        vec_norm = vec_dim / np.linalg.norm(vec_dim)
        vec_origin = (high_avg_embed + low_avg_embed) / 2
    
    elif args.model_method == 'regression':
        X = np.concatenate([high_embeddings, low_embeddings, neutral_embeddings])
        y = np.array([1] * len(high_embeddings) + [-1] * len(low_embeddings) + [0] * len(neutral_embeddings)).reshape(-1,1)
        reg = LinearRegression().fit(X,y)

    article_embed_paths = [os.path.join(relations_path, filename) for filename in os.listdir(relations_path) if path_keyword in filename]

    for article_embed_path in article_embed_paths:
        print(f"Getting index for {os.path.basename(article_embed_path)}")
        df = pd.read_csv(article_embed_path)
        df = df.rename(columns={'embed': 'embedding'})
        df = df[['publish_date', 'title', 'embedding']]
        df['publish_date'] = pd.to_datetime(df['publish_date'])
        df['embedding'] = df['embedding'].str.strip('[]').apply(lambda x: np.fromstring(x, sep=' '))
        df['index'] = df['embedding'].apply(lambda embed: reg.predict(embed.reshape(1,-1))[0][0])
        df = df[['publish_date', 'title', 'index']]
        df.to_csv(article_embed_path.replace(path_keyword, 'war_peace_axis'), index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--embed-method')
    parser.add_argument('--label-method')
    parser.add_argument('--model-method')
    args = parser.parse_args()

    main(args)