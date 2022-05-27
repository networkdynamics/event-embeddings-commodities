import argparse
import datetime
import operator
import os
import random

import numpy as np
import pandas as pd
import torch
import tqdm

TEACHER_FORCING_RATIO = 0.5
MAX_EPOCHS = 10000
EARLY_STOPPING_PATIENCE = 25
SEQUENCE_LENGTH = 100


class CommodityDataset(torch.utils.data.Dataset):
    def __init__(self, commodity, embed_suffix, days_ahead, seq_len):

        self.days_ahead = days_ahead
        self.seq_len = seq_len

        this_dir_path = os.path.dirname(os.path.abspath(__file__))
        commodity_dir = os.path.join(this_dir_path, '..', '..', 'data', 'commodity_data')

        commodity_price_path = os.path.join(commodity_dir, f"{commodity}.csv")
        commodity_article_path = os.path.join(commodity_dir, f"{commodity}_{embed_suffix}.csv")
        price_df = pd.read_csv(commodity_price_path)
        price_df = price_df.rename(columns={'Date': 'date', 'Close': 'close'})
        price_df = price_df[['date', 'close']]
        price_df = price_df.dropna()
        price_df = price_df.sort_values('date')
        price_df['last_date'] = price_df['date'].shift(1)
        price_df.at[0, 'last_date'] = datetime.datetime.strptime(price_df.loc[0, 'date'], '%Y-%m-%d').date() - datetime.timedelta(days=1)
        price_df['date'] = pd.to_datetime(price_df['date'])
        price_df['last_date'] = pd.to_datetime(price_df['last_date'])


        article_df = pd.read_csv(commodity_article_path)
        article_df = article_df[['title', 'publish_date', 'embedding']]
        article_df['publish_date'] = pd.to_datetime(article_df['publish_date'])
        article_df['embedding'] = article_df['embedding'].str.strip('[]').apply(lambda x: np.fromstring(x, sep=' '))

        self.embedding_size = len(article_df['embedding'].iloc[0])

        # group articles by day
        article_df = article_df.groupby('publish_date').agg(list).reset_index()

        # group articles by next trading day
        df_merge = price_df.merge(article_df, how='cross')
        df_merge = df_merge.query('publish_date > last_date and publish_date <= date')
        df = price_df.merge(df_merge, on=['last_date','date'], how='left')
        df = df.rename(columns={'close_x': 'close'})
        df = df[['date', 'close', 'title', 'embedding']]
        df = df.groupby(['date', 'close']).agg(sum).reset_index()

        # normalize price
        df['close'] = (df['close'] - df['close'].mean()) / df['close'].std()
        df['last_close'] = df['close'].shift(self.days_ahead)
        df = df.iloc[self.days_ahead:]

        # perform padding
        max_articles = df[df['embedding'] != 0]['embedding'].apply(len).max()

        def pad_embeddings(embeddings):
            padded = np.zeros((max_articles, self.embedding_size), dtype=float)
            if embeddings != 0:
                for idx in range(len(embeddings)):
                    padded[idx,:] = embeddings[idx][:]
            return padded


        def create_attention_mask(embeddings):
            attention_mask = np.zeros((max_articles,), dtype=int)
            if embeddings != 0:
                attention_mask[:len(embeddings)] = 1
            return attention_mask

        df['padded_embedding'] = df['embedding'].apply(pad_embeddings)
        df['attention_mask'] = df['embedding'].apply(create_attention_mask)

        self.df = df

    def __len__(self):
        return len(self.df) - self.seq_len + 1

    def __getitem__(self, idx):
        sequence = self.df.iloc[idx:idx+self.seq_len]

        data_seq = {
            'inputs': torch.tensor(sequence['last_close'].values).float(), 
            'targets': torch.tensor(sequence['close'].values).float(), 
            'encoder_outputs': torch.tensor(np.stack(sequence['padded_embedding'].values)).float(),
            'attention_mask': torch.tensor(np.stack(sequence['attention_mask'].values)).float()
        }
        return data_seq


class AttnDecoderRNN(torch.nn.Module):
    def __init__(self, embedding_size, hidden_size, combine='attn', dropout_p=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.combine = combine
        self.dropout_p = dropout_p

        self.attn = torch.nn.Linear(embedding_size + hidden_size + 1, 1)
        self.attn_combine = torch.nn.Linear(embedding_size + 1, hidden_size)
        self.dropout = torch.nn.Dropout(self.dropout_p)
        self.gru = torch.nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, input, hidden, encoder_outputs, attention_mask):

        if self.combine == 'attn':
            num_attn_points = encoder_outputs.shape[1]
            attn_params = torch.cat((encoder_outputs, torch.permute(hidden, (1,0,2)).expand(-1, num_attn_points, -1), input.view(-1, 1, 1).expand(-1, num_attn_points, -1)), 2)
            attn_weights = self.attn(attn_params)
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=1)
            attn_weights = attn_weights.squeeze(2) * attention_mask
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        elif self.combine == 'avg':
            attn_applied = torch.mean(encoder_outputs, 1)

        output = torch.cat((input.unsqueeze(1), attn_applied.squeeze(1)), 1)
        output = self.attn_combine(output)

        output = torch.nn.functional.relu(output)
        output = self.dropout(output)

        output, hidden = self.gru(output.unsqueeze(1), hidden)

        output = torch.nn.functional.relu(output)
        output = self.dropout(output)

        output = self.out(output).squeeze(2).squeeze(1)
        return output, hidden, attn_weights

    def init_hidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


def train_model(encoder_outputs, attention_masks, inputs, targets, decoder, criterion, device, days_ahead):
    
    batch_size = targets.shape[0]
    target_length = targets.shape[1]

    loss = 0

    decoder_hidden = decoder.init_hidden(batch_size, device)

    use_teacher_forcing = random.random() < TEACHER_FORCING_RATIO

    # TODO work this out for multiple days ahead forecasting
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for idx in range(target_length):
            decoder_input = inputs[:, idx]  # Teacher forcing
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs[:,idx,:,:], attention_masks[:,idx,:])
            loss += criterion(decoder_output, targets[:,idx])

    else:
        # Without teacher forcing: use its own predictions as the next input
        decoder_input = inputs[:, 0]
        for idx in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs[:,idx,:,:], attention_masks[:,idx,:])
            
            if idx < days_ahead:
                decoder_input = inputs[:, idx+1]
            else:
                decoder_input = decoder_output.detach() # detach from history as input

            loss += criterion(decoder_output, targets[:,idx])

    return loss

def train(model, train_data, val_data, device, checkpoint_path, resume, days_ahead, seq_len):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    criterion = torch.nn.MSELoss()
    model.train()

    min_val_loss = 999999
    epochs_since_best = 0

    if resume:
        model.load_state_dict(torch.load(checkpoint_path))

    for epoch in range(MAX_EPOCHS):

        # train on training set
        train_loss = 0
        progress_bar_data = tqdm.tqdm(enumerate(train_data), total=len(train_data))
        for batch_idx, batch in progress_bar_data:
            encoder_outputs = batch['encoder_outputs'].to(device)
            attention_masks = batch['attention_mask'].to(device)
            inputs = batch['inputs'].to(device)
            targets = batch['targets'].to(device)
            
            optimizer.zero_grad()
            loss = train_model(encoder_outputs, attention_masks, inputs, targets, model, criterion, device, days_ahead)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item() / targets.shape[1]
            progress_bar_data.set_description(f"Current Loss: {batch_loss:.4f}")
            train_loss += batch_loss
        
        train_loss /= len(train_data)

        # trail on validation set
        val_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_data):
                encoder_outputs = batch['encoder_outputs'].to(device)
                attention_masks = batch['attention_mask'].to(device)
                inputs = batch['inputs'].to(device)
                targets = batch['targets'].to(device)

                loss = train_model(encoder_outputs, attention_masks, inputs, targets, model, criterion, device, days_ahead)
                batch_loss = loss.item() / targets.shape[1]
                val_loss += batch_loss

        val_loss /= len(val_data)

        # potentially update learning rate
        scheduler.step(val_loss)

        # save best model so far
        if val_loss < min_val_loss:
            # checkpoint model
            torch.save(model.state_dict(), checkpoint_path)
            min_val_loss = val_loss
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        # early stopping
        if epochs_since_best > EARLY_STOPPING_PATIENCE:
            print(f"Best Val Loss: {min_val_loss:.4f}")
            break

        print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

def test_model(encoder_outputs, attention_mask, target_tensor, decoder, criterion, device, days_ahead):
    
    target_length = target_tensor.size(0)

    loss = 0

    decoder_input = torch.tensor([0], device=device)
    decoder_hidden = decoder.init_hidden()

    use_teacher_forcing = random.random() < TEACHER_FORCING_RATIO

    # TODO work this out for multiple days ahead forecasting
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for idx in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs[idx])
            loss += criterion(decoder_output, target_tensor[idx])
            decoder_input = target_tensor[idx]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for idx in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs[idx])
            decoder_input = decoder_output.detach() # detach from history as input

            loss += criterion(decoder_output, target_tensor[idx])

    return loss.item() / target_length

def test(model, test_data, device, checkpoint_path, days_ahead, seq_len):

    criterion = torch.nn.MSELoss()
    model.eval()

    model.load_state_dict(torch.load(checkpoint_path))

        # train on training set
    test_loss = 0
    progress_bar_data = tqdm.tqdm(enumerate(test_data), total=len(test_data))
    with torch.no_grad():
        for batch_idx, batch in progress_bar_data:
            encoder_outputs = batch['encoder_outputs'].to(device)
            targets = batch['targets'].to(device)
            
            batch_loss = test_model(encoder_outputs, targets, model, criterion, device, days_ahead, seq_len)
            test_loss += batch_loss
        
    test_loss /= len(test_data)

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
    seq_len = 50
    hidden_size = 5

    train_data, val_data, test_data, embedding_size = load_data(args.commodity, args.embed_suffix, args.batch_size, args.days_ahead, seq_len)

    if not os.path.exists(os.path.dirname(args.checkpoint_path)):
        os.makedirs(os.path.dirname(args.checkpoint_path))

    embedding_size = embedding_size
    model = AttnDecoderRNN(embedding_size, hidden_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if args.mode == 'train':
        train(model, train_data, val_data, device, args.checkpoint_path, args.resume, args.days_ahead, seq_len)
    elif args.mode == 'test':
        test(model, test_data, device, args.checkpoint_path, args.days_ahead, seq_len)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--commodity')
    parser.add_argument('--embed-suffix')
    parser.add_argument('--mode')
    parser.add_argument('--resume')
    parser.add_argument('--checkpoint-path')
    parser.add_argument('--days-ahead', type=int, default=1)
    args = parser.parse_args()

    main(args)