import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import transformers
import tqdm

import info_nce

TEACHER_FORCING_RATIO = 0.5
MAX_EPOCHS = 10000
EARLY_STOPPING_PATIENCE = 25
SEQUENCE_LENGTH = 100


class ArticleGraphDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir):

        dataset_files = os.listdir(dataset_dir)
        for edge_file in [dataset_file for dataset_file in dataset_files if 'edgelist' in dataset_file]:    
            print(f"Loading graph from file {edge_file}")

            edge_filepath = os.path.join(dataset_dir, edge_file)
            nx_G = read_graph(edge_filepath)
            G = random_walks.Graph(nx_G, args.directed, args.p, args.q)
            print("Preprocess transition probabilities")
            G.preprocess_transition_probs()
            print("Simulate walks")
            triplets = get_triplets(args.num_walks, args.walk_length)
            
            print("Map news")
            map_filepath = edge_filepath.replace("edges.csv", "_articles.csv")
            article_walks = map_news(walks, map_filepath)

            all_article_walks += article_walks

        self.df = df

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        sequence = self.df.iloc[idx:idx+self.seq_len]

        data_seq = {
            'inputs': torch.tensor(sequence['last_close'].values).float(), 
            'targets': torch.tensor(sequence['close'].values).float(), 
            'encoder_outputs': torch.tensor(np.stack(sequence['padded_embedding'].values)).float(),
            'attention_mask': torch.tensor(np.stack(sequence['attention_mask'].values)).float()
        }
        return data_seq


class LongformerEmbed(torch.nn.Module):
    def __init__(self, embedding_size, hidden_size, dropout_p=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p

        self.model = transformers.LongformerModel.from_pretrained("allenai/longformer-base-4096")

        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(self.dropout_p)
        self.out_proj = torch.nn.Linear(hidden_size, embedding_size)

    def forward(self, input_ids, attention_mask):

        outputs = self.model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        hidden_states = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.out_proj(hidden_states)
        return output


def train(model, train_data, val_data, device, checkpoint_path, resume, days_ahead, seq_len):

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
        progress_bar_data = tqdm.tqdm(enumerate(train_data), total=len(train_data))
        for batch_idx, batch in progress_bar_data:
            input_ids = batch['input_ids'].to(device)
            input_attention_masks = batch['input_attention_mask'].to(device)
            target_ids = batch['target_ids'].to(device)
            target_attention_masks = batch['target_attention_mask'].to(device)
            negative_ids = batch['negative_ids'].to(device)
            negative_attention_masks = batch['negative_attention_mask'].to(device)
            
            optimizer.zero_grad()
            input_embed = model(input_ids, input_attention_masks)
            target_embed = model(target_ids, target_attention_masks)

            # TODO reshape
            negative_embed = model(negative_ids, negative_attention_masks)
            # TODO reshape back

            loss = criterion(input_embed, target_embed, negative_embed)

            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            progress_bar_data.set_description(f"Current Loss: {batch_loss:.4f}")
            train_loss += batch_loss
        
        train_loss /= len(train_data)

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


def load_data(commodity, embed_suffix, batch_size, days_ahead, seq_len):
    dataset = CommodityDataset(commodity, embed_suffix, days_ahead, seq_len)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    indices = list(range(len(dataset)))
    
    train_dataset = torch.utils.data.Subset(dataset, indices[:train_size])
    val_dataset = torch.utils.data.Subset(dataset, indices[train_size:train_size + val_size])
    test_dataset = torch.utils.data.Subset(dataset, indices[-test_size:])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader, dataset.embedding_size

def main(args):
    seq_len = 100
    hidden_size = 4

    train_data, val_data, test_data, embedding_size = load_data(args.commodity, args.embed_suffix, args.batch_size, args.days_ahead, seq_len)

    if not os.path.exists(os.path.dirname(args.checkpoint_path)):
        os.makedirs(os.path.dirname(args.checkpoint_path))

    embedding_size = embedding_size
    model = AttnDecoderRNN(embedding_size, hidden_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    train(model, train_data, val_data, device, args.checkpoint_path, args.resume, args.days_ahead, seq_len)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--commodity')
    parser.add_argument('--embed-suffix')
    parser.add_argument('--resume')
    parser.add_argument('--checkpoint-path')
    parser.add_argument('--days-ahead', type=int, default=1)
    args = parser.parse_args()

    main(args)