
import os
import random

import networkx as nx
import numpy as np
import pandas as pd
import torch
import tqdm
import transformers

NUM_NEGATIVE = 2
MAX_TOKENS = 512
MIN_COMMON_ENTITIES = 3


class ChunkDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.cum_lengths = None
        self.processed_paths = None
        self.process()

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

class ArticleGraphDataset(ChunkDataset):

    def process(self):

        tokenizer = transformers.RobertaTokenizerFast.from_pretrained("distilroberta-base")

        self.cum_lengths = []
        self.processed_paths = []
        dataset_files = os.listdir(self.dataset_dir)
        for edge_file in [dataset_file for dataset_file in dataset_files if 'edges' in dataset_file]:

            time_period = edge_file[:4]
            processed_path = os.path.join(self.dataset_dir, f"{time_period}.pt")
            self.processed_paths.append(processed_path)
            if os.path.exists(processed_path):  
                continue

            print(f"Loading graph from file {edge_file}")

            triplets = []

            edge_filepath = os.path.join(self.dataset_dir, edge_file)
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

class ConcatDatasetChunkSampler:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        dataset_indices = []
        for dataset_idx in range(len(self.dataset.datasets)):
            cum_lengths = self.dataset.datasets[dataset_idx].cum_lengths
            chunk_order = torch.randperm(len(cum_lengths))
            chunks = [0] + cum_lengths
            dataset_indices.append([])
            for chunk_idx in chunk_order:
                dataset_cum_size = 0 if dataset_idx == 0 else self.dataset.cumulative_sizes[dataset_idx - 1]
                indices = torch.randperm(chunks[chunk_idx + 1] - chunks[chunk_idx]) + chunks[chunk_idx] + dataset_cum_size
                dataset_indices[dataset_idx] += indices.tolist()

        indices_options = list(range(len(dataset_indices)))
        indices_idx = [0] * len(dataset_indices)
        indices_left = np.array([len(indices) for indices in dataset_indices])
        indices = []
        while sum(indices_left) > 0:
            idx = np.random.choice(indices_options, p=indices_left / np.sum(indices_left))
            
            indices.append(dataset_indices[idx][indices_idx[idx]])
            indices_idx[idx] += 1
            indices_left[idx] -= 1

        assert len(indices) == sum([len(set_indices) for set_indices in dataset_indices])

        yield from indices

def get_top_n_valid_predecessors(graph, node, n):
    predecessors = graph.predecessors(node)
    predec_weights = []
    for predecessor in predecessors:
        num_entities = graph[predecessor][node]['entities']
        if num_entities >= MIN_COMMON_ENTITIES:
            predec_weights.append((predecessor, num_entities))

    sorted_predecessors = [predec[0] for predec in sorted(predec_weights, key=lambda x: x[1], reverse=True)]

    num_predecessors = graph.in_degree(node)
    return sorted_predecessors[:min(n, num_predecessors)]


class ArticleContextDataset(ChunkDataset):
    def process(self):

        tokenizer = transformers.RobertaTokenizerFast.from_pretrained("distilroberta-base")

        node_file_names = [file for file in os.listdir(self.dataset_dir) if 'nodes' in file]

        self.cum_lengths = []
        self.processed_paths = []

        for node_file_name in node_file_names:

            time_period = node_file_name[:2]
            processed_path = os.path.join(self.dataset_dir, f"{time_period}.pt")
            self.processed_paths.append(processed_path)
            if os.path.exists(processed_path):  
                continue

            edge_file_name = node_file_name.replace('nodes', 'edges')
            
            # load graph from file
            nodes_df = pd.read_csv(os.path.join(self.dataset_dir, node_file_name))
            edges_df = pd.read_csv(os.path.join(self.dataset_dir, edge_file_name))

            # drop nan rows
            nodes_df = nodes_df.dropna()

            # drop duplicate titles
            nodes_df = nodes_df.drop_duplicates(subset='title')

            graph = nx.DiGraph()

            node_mapping = {}
            # add nodes from dataframe
            print(f"Loading nodes into graph from {node_file_name}")
            for idx, node_row in tqdm.tqdm(nodes_df.iterrows(), total=len(nodes_df)):
                node_mapping[node_row['id']] = idx

                title = node_row['title']
                text = node_row['text']
                all_text = title + '. ' + text
                all_text = clean_text(all_text)

                graph.add_node(idx, node_text=all_text)

            # add edges from dataframe
            print(f"Loading edges into graph from {edge_file_name}")
            for idx, edge_row in tqdm.tqdm(edges_df.iterrows(), total=len(edges_df)):
                if edge_row['old_id'] not in node_mapping or edge_row['new_id'] not in node_mapping:
                    continue

                old_id = node_mapping[edge_row['old_id']]
                new_id = node_mapping[edge_row['new_id']]

                if graph.has_edge(old_id, new_id):
                    graph[old_id][new_id]['entities'] += 1
                else:
                    graph.add_edge(old_id, new_id, entities=1)

            # process network into torch compat shape
            triplets = []

            print('Create context/target pairs from graph')
            for node, node_data in tqdm.tqdm(graph.nodes(data=True), total=graph.number_of_nodes()):
                if graph.in_degree(node) == 0:
                    continue

                predecessors = get_top_n_valid_predecessors(graph, node, 1)
                if len(predecessors) == 0:
                    continue

                context_node = predecessors[0]

                data = {}

                node_text = node_data['node_text']

                tokens = tokenizer(node_text, padding='max_length', truncation=True, max_length=MAX_TOKENS)
                data['anchor_ids'] = torch.tensor(tokens.input_ids, dtype=torch.long)
                data['anchor_attention_mask'] = torch.tensor(tokens.attention_mask, dtype=torch.long)

                context_text = graph.nodes[context_node]['node_text']
                context_tokens = tokenizer(context_text, padding='max_length', truncation=True, max_length=MAX_TOKENS)
                
                data['positive_ids'] = torch.tensor(context_tokens.input_ids)
                data['positive_attention_mask'] = torch.tensor(context_tokens.attention_mask)

                negative_nodes = []
                max_tries = NUM_NEGATIVE * 3
                nodes = list(graph.nodes())
                tries = 0
                while len(negative_nodes) < NUM_NEGATIVE and tries < max_tries:
                    tries += 1
                    negative_node = random.choice(nodes)
                    if negative_node == node:
                        continue

                    if graph.has_edge(node, negative_node):
                        continue

                    if graph.has_edge(negative_node, node):
                        continue

                    negative_nodes.append(negative_node)

                if len(negative_nodes) != NUM_NEGATIVE:
                    continue

                negative_ids = []
                negative_attention_masks = []

                for negative_node in negative_nodes:
                    negative_text = graph.nodes[negative_node]['node_text']
                    negative_tokens = tokenizer(negative_text, padding='max_length', truncation=True, max_length=MAX_TOKENS)
                    negative_ids.append(torch.tensor(negative_tokens.input_ids))
                    negative_attention_masks.append(torch.tensor(negative_tokens.attention_mask))

                data['negative_ids'] = torch.stack(negative_ids)
                data['negative_attention_mask'] = torch.stack(negative_attention_masks)
                
                triplets.append(data)

            torch.save(triplets, processed_path)
            new_cum_length = len(triplets) + self.cum_lengths[-1] if self.cum_lengths else len(triplets)
            self.cum_lengths.append(new_cum_length)

        self.current_processed_path = None


def load_data(graph_dataset_path, context_dataset_path, batch_size):
    graph_dataset = ArticleGraphDataset(graph_dataset_path)
    context_dataset = ArticleContextDataset(context_dataset_path)
    dataset = torch.utils.data.ConcatDataset([graph_dataset, context_dataset])

    indices = list(range(len(dataset)))
    subset = torch.utils.data.Subset(dataset, indices[:])

    sampler = ConcatDatasetChunkSampler(dataset)
    dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, sampler=sampler)

    return dataloader