import argparse
import os

import info_nce
import networkx as nx
import numpy as np
import pandas as pd
from py import process
from regex import D
import torch
import transformers
import tqdm

import datasets

MAX_EPOCHS = 10000
EARLY_STOPPING_PATIENCE = 25
MAX_TOKENS = 512
NUM_NEGATIVE = 2
END_STRING_TOKEN_ID = 2


class LanguageModelEmbed(torch.nn.Module):
    def __init__(self, embed_size=128):
        super().__init__()

        config = transformers.RobertaConfig.from_pretrained("distilroberta-base")
        config.num_labels = embed_size
        self.model = transformers.RobertaForSequenceClassification.from_pretrained("distilroberta-base", config=config)

        for param in self.model.roberta.embeddings.parameters():
            param.requires_grad = False

        for param in self.model.roberta.encoder.layer[:-1].parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):

        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        #hidden_state = hidden_state[:, 0, :]  # take <s> token (equiv. to [CLS])
        logits = torch.nn.functional.normalize(logits, dim=-1)
        return logits


def train(model, data, device, checkpoint_path, resume):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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
            all_text = datasets.clean_text(all_text)
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

    model.eval()
    model.load_state_dict(torch.load(checkpoint_path))

    for article_path in article_paths:

        article_embed_path = article_path.replace('articles', 'lm_small_embed')
        if os.path.exists(article_embed_path):
            continue

        print(f"Embedding articles in {os.path.basename(article_path)}")

        articles_df = pd.read_csv(article_path)
        articles_df = articles_df[['text', 'title', 'publish_date', 'url']]
        dataloader = prep_articles_for_embed(articles_df, batch_size)

        embed_df = None
        progress_bar_data = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
        for batch_idx, batch in progress_bar_data:
            article_ids = batch['article_id']
            input_ids = batch['input_ids'].to(device)
            attention_masks = batch['attention_mask'].to(device)
            
            input_embed = model(input_ids, attention_masks)

            ids = article_ids.detach().numpy()
            embeds = input_embed.cpu().detach().numpy()

            batch_embed_df = pd.DataFrame([{'id': id, 'embed': embed} for id, embed in zip(ids, embeds)])
            if embed_df is not None:
                embed_df = pd.concat([embed_df, batch_embed_df])
            else:
                embed_df = batch_embed_df

        embed_df = embed_df.set_index('id', drop=True)
        article_embed_df = articles_df.join(embed_df)
        article_embed_df.to_csv(article_embed_path)




def main(args):

    if not os.path.exists(os.path.dirname(args.checkpoint_path)):
        os.makedirs(os.path.dirname(args.checkpoint_path))

    model = LanguageModelEmbed(embed_size=args.embed_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if args.mode == 'train':
        assert args.graph_dataset_path
        assert args.context_dataset_path
        dataloader = datasets.load_data(args.graph_dataset_path, args.context_dataset_path, args.batch_size)
        train(model, dataloader, device, args.checkpoint_path, args.resume)
    elif args.mode == 'embed':
        embed(model, args.dataset_path, args.batch_size, device, args.checkpoint_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'embed'], default='train')
    parser.add_argument('--graph-dataset-path')
    parser.add_argument('--context-dataset-path')
    parser.add_argument('--dataset-path')
    parser.add_argument('--embed-size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--checkpoint-path')
    args = parser.parse_args()

    main(args)