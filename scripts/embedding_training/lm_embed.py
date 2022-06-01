import argparse
import os
import random
from tkinter.tix import MAX

import networkx as nx
import numpy as np
import pandas as pd
from py import process
from regex import D
import torch
import transformers
import tqdm

import info_nce

MAX_EPOCHS = 10000
EARLY_STOPPING_PATIENCE = 25
MAX_TOKENS = 512
NUM_NEGATIVE = 2
END_STRING_TOKEN_ID = 2


def clean_text(txt):
    txt = txt.replace('Register now for FREE unlimited access to reuters.com Register', ' ') \
             .replace('\n', ' ') \
             .replace('Our Standards: The Thomson Reuters Trust Principles.', '')
    return txt

def get_triplet_ids(graph, node_ids):
    triplets = []
    for node_id in tqdm.tqdm(node_ids):

        if not graph.has_node(node_id):
            continue

        edges = graph.edges(node_id, data=True)
        neighbours = np.zeros(len(edges), dtype=int)
        neighbour_weights = np.zeros(len(edges))
        for id, (_, neighbour_id, attrs) in enumerate(edges):
            neighbours[id] = neighbour_id
            neighbour_weights[id] = attrs['weight']
        neighbour_weights = neighbour_weights / sum(neighbour_weights)
        word_id = np.random.choice(a=neighbours, p=neighbour_weights)

        next_edges = graph.edges(word_id, data=True)
        next_edges = [edge for edge in next_edges if edge[1] != node_id]
        neighbour_neighbours = np.zeros(len(next_edges), dtype=int)
        neighbour_neighbour_weights = np.zeros(len(next_edges))
        for id, (_, neighbour_id, attrs) in enumerate(next_edges):
            neighbour_neighbours[id] = neighbour_id
            neighbour_neighbour_weights[id] = attrs['weight']
        neighbour_neighbour_weights = neighbour_neighbour_weights / sum(neighbour_neighbour_weights)

        positive_id = np.random.choice(a=neighbour_neighbours, p=neighbour_neighbour_weights)

        neighbour_set = set(neighbours)
        negative_ids = []
        MAX_TRIES = NUM_NEGATIVE * 3
        tries = 0
        while len(negative_ids) < NUM_NEGATIVE and tries < MAX_TRIES:
            tries += 1
            negative_id = random.choice(node_ids)
            if negative_id == node_id:
                continue
            if not graph.has_node(negative_id):
                continue

            negative_neighbours = set(graph.neighbors(negative_id))
            if neighbour_set.isdisjoint(negative_neighbours):
                negative_ids.append(negative_id)

        if len(negative_ids) != NUM_NEGATIVE:
            continue

        triplets.append({'anchor_id': node_id, 'positive_id': positive_id, 'negative_ids': negative_ids, 'word_id': word_id})

    return triplets

class ArticleGraphDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir):

        tokenizer = transformers.RobertaTokenizerFast.from_pretrained("distilroberta-base")

        self.cum_lengths = []
        self.processed_paths = []
        dataset_files = os.listdir(dataset_dir)
        for edge_file in [dataset_file for dataset_file in dataset_files if 'edges' in dataset_file]:

            time_period = edge_file[:4]
            processed_path = os.path.join(dataset_dir, f"{time_period}.pt")
            self.processed_paths.append(processed_path)
            if os.path.exists(processed_path):  
                continue

            print(f"Loading graph from file {edge_file}")

            triplets = []

            edge_filepath = os.path.join(dataset_dir, edge_file)
            edges_df = pd.read_csv(edge_filepath)
            graph = nx.from_pandas_edgelist(edges_df, source='id1', target='id2', edge_attr='weight')

            article_filepath = edge_filepath.replace("edges.csv", "articles.csv")
            article_df = pd.read_csv(article_filepath)

            word_filepath = edge_filepath.replace("edges.csv", "words.csv")
            word_df = pd.read_csv(word_filepath)

            article_features = {}
            for index, row in article_df.iterrows():
                id = row['id']
                title = row['title']
                text = row['text']
                all_text = title + '. ' + text
                all_text = clean_text(all_text)
                tokenized = tokenizer(all_text, padding='max_length', truncation=True, max_length=MAX_TOKENS)

                article_features[id] = tokenized

            node_ids = article_df['id'].values.tolist()
            print("Generating triplets")
            triplet_ids = get_triplet_ids(graph, node_ids)
            
            for triplet_id in triplet_ids:

                word = word_df[word_df['id'] == triplet_id['word_id']]['word'].values[0]

                triplet = {
                    'anchor_ids': torch.tensor(article_features[triplet_id['anchor_id']].input_ids),
                    'anchor_attention_mask': torch.tensor(article_features[triplet_id['anchor_id']].attention_mask),
                    'positive_ids': torch.tensor(article_features[triplet_id['positive_id']].input_ids),
                    'positive_attention_mask': torch.tensor(article_features[triplet_id['positive_id']].input_ids),
                }
                triplet['negative_ids'] = torch.stack([torch.tensor(article_features[negative_id].input_ids) for negative_id in triplet_id['negative_ids']])
                triplet['negative_attention_mask'] = torch.stack([torch.tensor(article_features[negative_id].attention_mask) for negative_id in triplet_id['negative_ids']])
                
                triplets.append(triplet)

            torch.save(triplets, processed_path)
            new_cum_length = len(triplets) + self.cum_lengths[-1] if self.cum_lengths else len(triplets)
            self.cum_lengths.append(new_cum_length)

        self.current_processed_path = None

    def __len__(self):
        if not self.cum_lengths:
            for processed_path in self.processed_paths:
                triplets = torch.load(processed_path)
                new_cum_length = len(triplets) + self.cum_lengths[-1] if self.cum_lengths else len(triplets)
                self.cum_lengths.append(new_cum_length)

        return self.cum_lengths[-1]

    def __getitem__(self, idx):
        for path_idx, (start_chunk, end_chunk) in enumerate(zip([0] + self.cum_lengths[:-1], self.cum_lengths)):
            if idx >= start_chunk and idx < end_chunk:
                processed_path = self.processed_paths[path_idx]
                break

        if not self.current_processed_path == processed_path:
            self.current_triplets = torch.load(processed_path)
            self.current_processed_path = processed_path

        return self.current_triplets[idx - start_chunk]


class ChunkSampler:
    def __init__(self, cum_lengths):
        self.cum_lengths = cum_lengths

    def __len__(self):
        return self.cum_lengths[-1]

    def __iter__(self):
        chunk_order = torch.randperm(len(self.cum_lengths))
        chunks = [0] + self.cum_lengths
        for chunk_idx in chunk_order:
            indices = torch.randperm(chunks[chunk_idx + 1] - chunks[chunk_idx]) + chunks[chunk_idx]
            yield from indices.tolist()


class LanguageModelEmbed(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model = transformers.RobertaModel.from_pretrained("distilroberta-base")

        for param in self.model.embeddings.parameters():
            param.requires_grad = False

        for param in self.model.encoder.layer[:-2].parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):

        outputs = self.model(input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state
        hidden_state = hidden_state[:, 0, :]  # take <s> token (equiv. to [CLS])
        return hidden_state


def train(model, data, device, checkpoint_path, resume):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    criterion = info_nce.InfoNCE(negative_mode='paired')
    model.train()

    min_train_loss = 999999
    epochs_since_best = 0

    if resume:
        model.load_state_dict(torch.load(checkpoint_path))

    for epoch in range(MAX_EPOCHS):

        # train on training set
        train_loss = 0
        progress_bar_data = tqdm.tqdm(enumerate(data), total=len(data))
        for batch_idx, batch in progress_bar_data:
            anchor_ids = batch['anchor_ids'].to(device)
            anchor_attention_masks = batch['anchor_attention_mask'].to(device)
            positive_ids = batch['positive_ids'].to(device)
            positive_attention_masks = batch['positive_attention_mask'].to(device)
            negative_ids = batch['negative_ids'].to(device)
            negative_attention_masks = batch['negative_attention_mask'].to(device)
            
            optimizer.zero_grad()
            input_embed = model(anchor_ids, anchor_attention_masks)
            positive_embed = model(positive_ids, positive_attention_masks)

            negative_embeds = torch.zeros((negative_ids.shape[0], negative_ids.shape[1], positive_embed.shape[1])).to(device)
            for negative_idx in range(negative_ids.shape[1]):
                negative_embeds[:,negative_idx,:] = model(negative_ids[:,negative_idx,:].squeeze(1), negative_attention_masks[:,negative_idx,:].squeeze(1))
            
            loss = criterion(input_embed, positive_embed, negative_embeds)

            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            progress_bar_data.set_description(f"Current Loss: {batch_loss:.4f}")
            train_loss += batch_loss
        
        train_loss /= len(data)

        # potentially update learning rate
        scheduler.step(train_loss)

        # save best model so far
        if train_loss < min_train_loss:
            # checkpoint model
            torch.save(model.state_dict(), checkpoint_path)
            min_train_loss = train_loss
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        # early stopping
        if epochs_since_best > EARLY_STOPPING_PATIENCE:
            print(f"Best Train Loss: {min_train_loss:.4f}")
            break

        print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}")


class ArticleData(torch.utils.data.Dataset):
    def __init__(self, article_df):
        tokenizer = transformers.RobertaTokenizerFast.from_pretrained("distilroberta-base")
        self.articles_data = []
        for idx, row in article_df.iterrows():
            title = row['title']
            text = row['text']
            all_text = title + '. ' + text
            all_text = clean_text(all_text)
            tokenized = tokenizer(all_text, padding='max_length', truncation=True, max_length=MAX_TOKENS)

            article_data = {
                'article_id': idx,
                'text': all_text,
                'input_ids': torch.tensor(tokenized.input_ids),
                'attention_mask': torch.tensor(tokenized.attention_mask)
            }
            self.articles_data.append(article_data)

    def __len__(self):
        return len(self.articles_data)

    def __getitem__(self, idx):
        return self.articles_data[idx]


def prep_articles_for_embed(articles_df, batch_size):

    article_data = ArticleData(articles_df)

    indices = list(range(len(article_data)))
    subset = torch.utils.data.Subset(article_data, indices[:])
    dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size)

    return dataloader


def embed(model, data_path, batch_size, device, checkpoint_path):

    if os.path.isdir(data_path):
        article_paths = [os.path.join(data_path, file_name) for file_name in os.listdir(data_path) if 'articles' in file_name]
    elif os.path.isfile(data_path):
        article_paths = [data_path]

    for article_path in article_paths:

        article_embed_path = article_path.replace('articles', 'lm_embed')
        if os.path.exists(article_embed_path):
            continue

        print(f"Embedding articles in {os.path.basename(article_path)}")

        articles_df = pd.read_csv(article_path)
        articles_df = articles_df[['text', 'title', 'publish_date', 'url']]
        dataloader = prep_articles_for_embed(articles_df, batch_size)

        model.eval()
        model.load_state_dict(torch.load(checkpoint_path))

        embed_df = None
        progress_bar_data = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
        for batch_idx, batch in progress_bar_data:
            article_ids = batch['article_id']
            texts = batch['text']
            input_ids = batch['input_ids'].to(device)
            attention_masks = batch['attention_mask'].to(device)
            
            input_embed = model(input_ids, attention_masks)

            ids = article_ids.detach().numpy()
            embeds = input_embed.cpu().detach().numpy()

            batch_embed_df = pd.DataFrame([{'id': id, 'embed': embed, 'text': text} for id, embed, text in zip(ids, embeds, texts)])
            if embed_df is not None:
                embed_df = pd.concat([embed_df, batch_embed_df])
            else:
                embed_df = batch_embed_df

        embed_df = embed_df.set_index('id', drop=True)
        article_embed_df = articles_df.join(embed_df, lsuffix='l_', rsuffix='r_')
        article_embed_df.to_csv(article_embed_path)


def load_data(dataset_path, batch_size):
    dataset = ArticleGraphDataset(dataset_path)

    indices = list(range(len(dataset)))
    subset = torch.utils.data.Subset(dataset, indices[:])

    sampler = ChunkSampler(dataset.cum_lengths)
    dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, sampler=sampler)

    return dataloader

def main(args):

    if not os.path.exists(os.path.dirname(args.checkpoint_path)):
        os.makedirs(os.path.dirname(args.checkpoint_path))

    model = LanguageModelEmbed()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if args.mode == 'train':
        data = load_data(args.dataset_path, args.batch_size)
        train(model, data, device, args.checkpoint_path, args.resume)
    elif args.mode == 'embed':
        embed(model, args.dataset_path, args.batch_size, device, args.checkpoint_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'embed'], default='train')
    parser.add_argument('--dataset-path')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--checkpoint-path')
    args = parser.parse_args()

    main(args)