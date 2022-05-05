import argparse
import datetime
import operator
import os
import random

import pandas as pd
import torch
import tqdm

TEACHER_FORCING_RATIO = 0.5
MAX_EPOCHS = 10000
EARLY_STOPPING_PATIENCE = 25
SEQUENCE_LENGTH = 300


class CommodityDataset(torch.utils.data.Dataset):
    def __init__(self, commodity):
        this_dir_path = os.path.dirname(os.path.abspath(__file__))
        commodity_dir = os.path.join(this_dir_path, '..', '..', 'data', 'commodity_data')

        commodity_price_path = os.path.join(commodity_dir, f"mock_{commodity}.csv")
        commodity_article_path = os.path.join(commodity_dir, f"mock_{commodity}_articles.csv")
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

        self.embedding_size = article_df['embedding'].first()

        # group articles by day
        article_df = article_df.groupby('publish_date').agg(list).reset_index()

        # group articles by next trading day
        df_merge = price_df.merge(article_df, how='cross')
        df_merge = df_merge.query('publish_date > last_date and publish_date <= date')
        df = price_df.merge(df_merge, on=['last_date','date'], how='left').fillna()
        df = df.rename(columns={'close_x': 'close'})
        df = df[['date', 'close', 'title', 'embedding']]
        df = df.groupby(['date', 'close']).agg(sum).reset_index()
        df['last_close'] = df['close'].shift(1)
        df = df.iloc[1:]
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sequence = self.df.iloc[idx:idx+SEQUENCE_LENGTH]

        return {'inputs': sequence['last_close'].values, 'targets': sequence['close'].values, 'encoder_outputs': sequence['last_close'].values}


class AttnDecoderRNN(torch.nn.Module):
    def __init__(self, embedding_size, hidden_size, dropout_p=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p

        self.attn = torch.nn.Linear(embedding_size + hidden_size + 1, 1)
        self.attn_combine = torch.nn.Linear(embedding_size + 1, hidden_size)
        self.dropout = torch.nn.Dropout(self.dropout_p)
        self.gru = torch.nn.GRU(self.hidden_size, self.hidden_size)
        self.out = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, input, hidden, encoder_outputs):

        attn_weights = self.attn(torch.cat((encoder_outputs, hidden, input), 1))
        attn_weights = torch.functional.softmax(attn_weights, dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((input[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = torch.functional.relu(output)
        output, hidden = self.gru(output, hidden)

        output = torch.functional.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def train_model(encoder_outputs, inputs, targets, decoder, optimizer, criterion, device):
    
    optimizer.zero_grad()

    target_length = targets.size(0)

    loss = 0

    decoder_hidden = decoder.init_hidden()

    use_teacher_forcing = random.random() < TEACHER_FORCING_RATIO

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for idx in range(target_length):
            decoder_input = inputs[idx]  # Teacher forcing
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs[idx])
            loss += criterion(decoder_output, targets[idx])

    else:
        decoder_input = torch.tensor(inputs[0], device=device)
        # Without teacher forcing: use its own predictions as the next input
        for idx in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs[idx])
            decoder_input = decoder_output.detach() # detach from history as input

            loss += criterion(decoder_output, targets[idx])

    loss.backward()
    optimizer.step()

    return loss.item() / target_length

def train(model, train_data, val_data, device, checkpoint_path, resume):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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
            inputs = batch['inputs'].to(device)
            targets = batch['targets'].to(device)
            
            batch_loss = train_model(encoder_outputs, inputs, targets, model, optimizer, criterion, device)
            progress_bar_data.set_description(f"Current Loss: {batch_loss:.4f}")
            train_loss += batch_loss
        
        train_loss /= len(train_data)

        # trail on validation set
        val_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_data):
                encoder_outputs = batch['encoder_outputs'].to(device)
                inputs = batch['inputs'].to(device)
                targets = batch['targets'].to(device)

                batch_loss = train_model(encoder_outputs, inputs, targets, model, optimizer, criterion, device)
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
            break

        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

def test_model(encoder_outputs, target_tensor, decoder, criterion, device):
    
    target_length = target_tensor.size(0)

    loss = 0

    decoder_input = torch.tensor([0], device=device)
    decoder_hidden = decoder.init_hidden()

    use_teacher_forcing = random.random() < TEACHER_FORCING_RATIO

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

def test(model, test_data, device, checkpoint_path):

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
            
            batch_loss = train_model(encoder_outputs, targets, model, criterion, device)
            test_loss += batch_loss
        
    test_loss /= len(test_data)

def load_data(commodity, batch_size):
    dataset = CommodityDataset(commodity)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader, dataset.embedding_size

def main(args):
    train_data, val_data, test_data, embedding_size = load_data(args.commodity, args.batch_size)

    embedding_size = embedding_size
    hidden_size = 4
    model = AttnDecoderRNN(embedding_size, hidden_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.mode == 'train':
        train(model, train_data, val_data, device, args.checkpoint_path, args.resume)
    elif args.mode == 'test':
        test(model, test_data, device, args.checkpoint_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--commodity')
    parser.add_argument('--mode')
    parser.add_argument('--resume')
    parser.add_argument('--checkpoint-path')
    args = parser.parse_args()

    main(args)