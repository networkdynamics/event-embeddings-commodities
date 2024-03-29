import argparse
import datetime
import os
import random
import sqlite3

import numpy as np
import pandas as pd
import torch
import tqdm

TEACHER_FORCING_RATIO = 0.5
MAX_EPOCHS = 10000
EARLY_STOPPING_PATIENCE = 10
SEQUENCE_LENGTH = 100


class CommodityDataset(torch.utils.data.Dataset):
    def __init__(self, commodity, suffix, days_ahead, seq_len, target='dir'):

        assert target in ['price', 'diff', 'dir']
        self.target = target

        self.days_ahead = days_ahead
        self.seq_len = seq_len

        this_dir_path = os.path.dirname(os.path.abspath(__file__))
        commodity_dir = os.path.join(this_dir_path, '..', '..', 'data', 'commodity_data')

        commodity_price_path = os.path.join(commodity_dir, f"{commodity}.csv")
        price_df = pd.read_csv(commodity_price_path)
        price_df = price_df.rename(columns={'Date': 'date', 'Close': 'close', 'Open': 'open'})
        price_df = price_df[['date', 'open', 'close']]
        price_df = price_df.dropna()
        price_df = price_df.sort_values('date')
        price_df['last_date'] = price_df['date'].shift(1)
        price_df.at[0, 'last_date'] = datetime.datetime.strptime(price_df.loc[0, 'date'], '%Y-%m-%d').date() - datetime.timedelta(days=1)
        price_df['date'] = pd.to_datetime(price_df['date'])
        price_df['last_date'] = pd.to_datetime(price_df['last_date'])

        if suffix != 'constant':
            commodity_article_path = os.path.join(commodity_dir, f"{commodity}_{suffix}.csv")
            article_df = pd.read_csv(commodity_article_path)#, dtype={'publish_date': str, 'title': str, 'embedding': str, 'embed': str, 'sentiment': float})

            article_df = article_df.dropna()

            article_df = article_df.rename(columns={'embed': 'embedding'})
            self.feature = [feat_type for feat_type in ['sentiment', 'embedding'] if feat_type in article_df.columns][0]

            article_df = article_df[['title', 'publish_date', self.feature]]
            article_df['publish_date'] = pd.to_datetime(article_df['publish_date'])
        
            # group articles by next trading day
            #Make the db in memory
            conn = sqlite3.connect(':memory:')
            #write the tables
            article_df.to_sql('articles', conn, index=False)
            price_df.to_sql('price', conn, index=False)

            qry = f'''
                select  
                    date,
                    open,
                    close,
                    title,
                    {self.feature}
                from
                    articles join price on
                    publish_date > last_date and publish_date <= date
                '''
            df = pd.read_sql_query(qry, conn)

            if self.feature == 'embedding':
                df['embedding'] = df['embedding'].str.strip('[]').replace('\n', '').apply(lambda x: np.fromstring(x, sep=' '))
                self.feature_size = len(df['embedding'].iloc[0])

            elif self.feature == 'sentiment':
                df['sentiment'] = df['sentiment'].apply(lambda x: np.array([x]))
                self.feature_size = 1
        else:
            self.feature = None
            self.feature_size = 1
            df = price_df

        df = df.groupby(['date', 'open', 'close']).agg(list).reset_index()

        # TODO data leakage, only normalize in each split
        # normalize price
        df['norm_close'] = (df['close'] - df['close'].mean()) / df['close'].std()
        
        # get targets
        df['target_close'] = df['norm_close'].shift(-self.days_ahead)
        df['target_close_diff'] = df['norm_close'].shift(-self.days_ahead) - df['norm_close'].shift(1)
        df['target_close_dir'] = df['target_close_diff'].apply(lambda x: 1 if x >= 0 else 0)

        # day to day changes
        self.start_nulls = 2
        df['previous_close'] = df['norm_close'].shift(1)
        df['previous_close_diff'] = df['norm_close'].shift(1) - df['norm_close'].shift(2)
        df['previous_close_dir'] = df['previous_close_diff'].apply(lambda x: 1 if x >= 0 else 0)
        self.start_nulls = 1 if self.target == 'price' else 2
        
        if suffix != 'constant':
            # perform padding
            max_articles = df[df[self.feature] != 0][self.feature].apply(len).max()

            def pad_features(features):
                padded = np.zeros((max_articles, self.feature_size), dtype=float)
                if features != 0:
                    for idx in range(len(features)):
                        padded[idx,:] = features[idx][:]
                return padded


            def create_attention_mask(features):
                attention_mask = np.zeros((max_articles,), dtype=int)
                if features != 0:
                    attention_mask[:len(features)] = 1
                return attention_mask

            df[f'padded_{self.feature}'] = df[self.feature].apply(pad_features)
            df['attention_mask'] = df[self.feature].apply(create_attention_mask)

        self.df = df

    def __len__(self):
        return len(self.df) - self.seq_len + 1 - self.start_nulls - self.days_ahead

    def __getitem__(self, idx):
        sequence = self.df.iloc[idx + self.start_nulls:idx + self.start_nulls + self.seq_len]

        if self.target == 'price':
            inputs = torch.tensor(sequence['previous_close'].values).float()
            targets = torch.tensor(sequence['target_close'].values).float()
        elif self.target == 'diff':
            inputs = torch.tensor(sequence['previous_close_diff'].values).float()
            targets = torch.tensor(sequence['target_close_diff'].values).float()
        elif self.target == 'dir':
            inputs = torch.tensor(sequence['previous_close_dir'].values).float()
            targets = torch.tensor(sequence['target_close_dir'].values).float()

        data_seq = {
            'index': torch.tensor(sequence.index.values),
            'inputs': inputs, 
            'targets': targets
        }

        if self.feature:
            encoder_outputs = torch.tensor(np.stack(sequence[f'padded_{self.feature}'].values)).float()
            attention_mask = torch.tensor(np.stack(sequence['attention_mask'].values), dtype=int)
        else:
            encoder_outputs = torch.ones(*inputs.shape[:2], 1, 1)
            attention_mask = torch.ones(*inputs.shape[:2], 1)
        
        data_seq['encoder_outputs'] = encoder_outputs
        data_seq['attention_mask'] = attention_mask

        return data_seq

class ConstantModel:
    def __init__(self, target):
        self.target = target

    def __call__(self, input, hidden, encoder_outputs, attention_mask):
        if self.target == 'price':
            return input, None, None
        elif self.target == 'diff':
            return torch.zeros(input.shape, device=input.device), None, None

class AttnDecoderRNN(torch.nn.Module):
    def __init__(self, feature_size, hidden_size, combine='attn', dropout_p=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.combine = combine
        self.dropout_p = dropout_p

        self.attn = torch.nn.Linear(feature_size + hidden_size + 1, 1)
        self.attn_combine = torch.nn.Linear(feature_size + 1, hidden_size)
        self.dropout = torch.nn.Dropout(self.dropout_p)
        self.gru = torch.nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, input, hidden, encoder_outputs, attention_mask):

        if self.combine == 'attn':
            num_attn_points = encoder_outputs.shape[1]
            attn_params = torch.cat((encoder_outputs, torch.permute(hidden, (1,0,2)).expand(-1, num_attn_points, -1), input.view(-1, 1, 1).expand(-1, num_attn_points, -1)), 2)
            attn_weights = self.attn(attn_params)
            attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(2) == 0, -1e10)
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=1)
            attn_applied = torch.bmm(attn_weights.transpose(1,2), encoder_outputs).squeeze(1)
        elif self.combine == 'avg':
            denom = torch.sum(attention_mask, -1, keepdim=True)
            denom[denom == 0] = 1
            attn_applied = torch.sum(encoder_outputs * attention_mask.unsqueeze(-1), dim=1) / denom
            attn_weights = None

        output = torch.cat((input.unsqueeze(1), attn_applied), 1)
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


class AttnDecoderRNN(torch.nn.Module):
    def __init__(self, feature_size, hidden_size, combine='attn', dropout_p=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.combine = combine
        self.dropout_p = dropout_p

        self.attn = torch.nn.Linear(feature_size + hidden_size + 1, 1)
        self.attn_combine = torch.nn.Linear(feature_size + 1, hidden_size)
        self.dropout = torch.nn.Dropout(self.dropout_p)
        self.gru = torch.nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, input, hidden, encoder_outputs, attention_mask):

        if self.combine == 'attn':
            num_attn_points = encoder_outputs.shape[1]
            attn_params = torch.cat((encoder_outputs, torch.permute(hidden, (1,0,2)).expand(-1, num_attn_points, -1), input.view(-1, 1, 1).expand(-1, num_attn_points, -1)), 2)
            attn_weights = self.attn(attn_params)
            attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(2) == 0, -1e10)
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=1)
            attn_applied = torch.bmm(attn_weights.transpose(1,2), encoder_outputs).squeeze(1)
        elif self.combine == 'avg':
            denom = torch.sum(attention_mask, -1, keepdim=True)
            denom[denom == 0] = 1
            attn_applied = torch.sum(encoder_outputs * attention_mask.unsqueeze(-1), dim=1) / denom
            attn_weights = None

        output = torch.cat((input.unsqueeze(1), attn_applied), 1)
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
        outputs = torch.zeros(targets.shape).float().to(device)
        for idx in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs[:,idx,:,:], attention_masks[:,idx,:])
            
            outputs[:, idx] = decoder_output.detach() # detach from history as input
            
            if idx < target_length - 1:
                if idx < days_ahead:
                    decoder_input = inputs[:, idx+1]
                else:
                    decoder_input = outputs[:, idx - days_ahead]

            loss += criterion(decoder_output, targets[:,idx])

    return loss


def train(model, train_data, val_data, device, checkpoint_path, resume, days_ahead, target, metric):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    if target == 'price' or target == 'diff':
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.BCELoss()

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

                batch_loss = test_model(encoder_outputs, attention_masks, inputs, targets, model, criterion, device, target, metric)
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

def test_model(encoder_outputs, attention_masks, inputs, targets, decoder, criterion, device, target, metric):
    
    batch_size = targets.shape[0]
    target_length = targets.shape[1]

    if target == 'dir':
        loss = torch.zeros((batch_size)).to(device)
    else:
        loss = 0

    if isinstance(decoder, torch.nn.Module):
        decoder_hidden = decoder.init_hidden(batch_size, device)
    else:
        decoder_hidden = None

    for idx in range(target_length):
        decoder_input = inputs[:, idx]  # Teacher forcing
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs[:,idx,:,:], attention_masks[:,idx,:])

        if metric == 'all':
            loss += criterion(decoder_output, targets[:,idx])
        elif metric == 'last':
            if idx == target_length - 1:
                loss += criterion(decoder_output, targets[:,idx])

    if target == 'dir':
        return float(torch.mean(loss / target_length))
    else:
        if metric == 'all':
            return loss.item() / target_length
        elif metric == 'last':
            return loss.item()

def test(model, test_data, device, checkpoint_path, target, metric):

    if target == 'price' or target == 'diff':
        criterion = torch.nn.MSELoss()
    else:
        def criterion(predicted, target):
            predicted_dir = (predicted >= 0.5).float()
            return 1 - torch.abs(predicted_dir - target)

    if isinstance(model, torch.nn.Module):
        model.eval()
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        # train on training set
    test_loss = 0
    progress_bar_data = tqdm.tqdm(enumerate(test_data), total=len(test_data))
    with torch.no_grad():
        for batch_idx, batch in progress_bar_data:
            encoder_outputs = batch['encoder_outputs'].to(device)
            attention_masks = batch['attention_mask'].to(device)
            inputs = batch['inputs'].to(device)
            targets = batch['targets'].to(device)
            
            batch_loss = test_model(encoder_outputs, attention_masks, inputs, targets, model, criterion, device, target, metric)
            test_loss += batch_loss
        
    test_loss /= len(test_data)
    print(f"Test Loss: {test_loss}")
    return test_loss



def load_data(commodity, suffix, batch_size, days_ahead, seq_len, target, split=[0.7, 0.15, 0.15]):
    dataset = CommodityDataset(commodity, suffix, days_ahead, seq_len, target)

    train_size = int(split[0] * len(dataset))
    val_size = int(split[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    indices = list(range(len(dataset)))
    
    train_dataset = torch.utils.data.Subset(dataset, indices[:train_size])
    val_dataset = torch.utils.data.Subset(dataset, indices[train_size:train_size + val_size])
    test_dataset = torch.utils.data.Subset(dataset, indices[-test_size:])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader, dataset.feature_size

def main(args):

    train_data, val_data, test_data, feature_size = load_data(args.commodity, args.suffix, args.batch_size, args.days_ahead, args.seq_len, args.target)

    if not os.path.exists(os.path.dirname(args.checkpoint_path)):
        os.makedirs(os.path.dirname(args.checkpoint_path))

    model = AttnDecoderRNN(feature_size, args.hidden_size, combine=args.combine)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if args.mode == 'train':
        train(model, train_data, val_data, device, args.checkpoint_path, args.resume, args.days_ahead, args.target, args.metric)
    elif args.mode == 'test':
        test(model, test_data, device, args.checkpoint_path, args.target, args.metric)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--combine', choices=['attn', 'avg'], default='attn')
    parser.add_argument('--commodity')
    parser.add_argument('--suffix')
    parser.add_argument('--mode', choices=['test', 'train'])
    parser.add_argument('--resume')
    parser.add_argument('--checkpoint-path')
    parser.add_argument('--target', default='dir')
    parser.add_argument('--days-ahead', type=int, default=1)
    parser.add_argument('--seq-len', type=int, default=50)
    parser.add_argument('--hidden-size', type=int, default=5)
    parser.add_argument('--metric', choices=['all', 'last'], default='last')
    args = parser.parse_args()

    main(args)