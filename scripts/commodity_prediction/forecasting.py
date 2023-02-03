import argparse
import datetime
import numpy as np
import os
import sqlite3

# imports for training
import lightning as pl
import pytorch_lightning.callbacks as plcb
import pandas as pd
import pytorch_forecasting as pyforecast


def get_datasets(commodity, suffix, days_ahead, seq_len, split=[0.7, 0.15, 0.15]):
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
        feature = [feat_type for feat_type in ['sentiment', 'embedding'] if feat_type in article_df.columns][0]

        article_df = article_df[['title', 'publish_date', feature]]
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
                {feature}
            from
                articles join price on
                publish_date > last_date and publish_date <= date
            '''
        df = pd.read_sql_query(qry, conn)

        if feature == 'embedding':
            df['embedding'] = df['embedding'].str.strip('[]').replace('\n', '').apply(lambda x: np.fromstring(x, sep=' '))
            feature_size = len(df['embedding'].iloc[0])

        elif feature == 'sentiment':
            df['sentiment'] = df['sentiment'].apply(lambda x: np.array([x]))
            feature_size = 1
    else:
        feature = None
        feature_size = 1
        df = price_df

    df = df.groupby(['date', 'open', 'close']).agg(list).reset_index()

    # # get targets
    # df['target_close'] = df['close'].shift(-days_ahead)
    # df['target_close_diff'] = df['close'].shift(-days_ahead) - df['close'].shift(1)
    # df['target_close_dir'] = df['target_close_diff'].apply(lambda x: 1 if x >= 0 else 0)

    # # day to day changes
    # start_nulls = 2
    # df['previous_close'] = df['close'].shift(1)
    # df['previous_close_diff'] = df['close'].shift(1) - df['close'].shift(2)
    # df['previous_close_dir'] = df['previous_close_diff'].apply(lambda x: 1 if x >= 0 else 0)
    # start_nulls = 1 if target == 'price' else 2
    
    if suffix != 'constant':
        # perform padding
        max_articles = df[df[feature] != 0][feature].apply(len).max()

        def pad_features(features):
            padded = np.zeros((max_articles, feature_size), dtype=float)
            if features != 0:
                for idx in range(len(features)):
                    padded[idx,:] = features[idx][:]
            return padded


        def create_attention_mask(features):
            attention_mask = np.zeros((max_articles,), dtype=int)
            if features != 0:
                attention_mask[:len(features)] = 1
            return attention_mask

        feature_col = f'padded_{feature}'
        attention_col = 'attention_mask'
        df[feature_col] = df[feature].apply(pad_features)
        df[attention_col] = df[feature].apply(create_attention_mask)

    dataset_size = len(df)
    train_size = int(split[0] * dataset_size)
    val_size = int(split[1] * dataset_size)

    # load data: this is pandas dataframe with at least a column for
    # * the target (what you want to predict)
    # * the timeseries ID (which should be a unique string to identify each timeseries)
    # * the time of the observation (which should be a monotonically increasing integer)
    
    df = df.reset_index()
    time_idx = 'index'
    future_col = 'close'
    group_col = 'group'
    df[group_col] = 'future'
    training = pyforecast.TimeSeriesDataSet(
        df[df[time_idx] < train_size],
        time_idx=time_idx,  # column name of time of observation
        target=future_col,  # column name of target to predict
        group_ids=[group_col],  # column name(s) for timeseries IDs
        max_encoder_length=seq_len,  # how much history to use
        max_prediction_length=days_ahead,  # how far to predict into future
        # covariates known and unknown in the future to inform prediction
        time_varying_unknown_reals=[feature_col, attention_col],
    )

    # create validation dataset using the same normalization techniques as for the training dataset
    validation = pyforecast.TimeSeriesDataSet.from_dataset(training, df[df[time_idx] < train_size + val_size], min_prediction_idx=training.index.time.max() + 1, stop_randomization=True)
    testset = pyforecast.TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=validation.index.time.max() + 1, stop_randomization=True)
    return training, validation, testset

def get_dataloaders(training, validation, testset, batch_size):
    # convert datasets to dataloaders for training
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=2)
    test_dataloader = testset.to_dataloader(train=False, batch_size=batch_size, num_workers=2)
    return train_dataloader, val_dataloader, test_dataloader

def get_mlp_model(training):
    # define network to train - the architecture is mostly inferred from the dataset, so that only a few hyperparameters have to be set by the user
    mlp = pyforecast.DecoderMLP.from_dataset(
        # dataset
        training,
        # architecture hyperparameters
        hidden_size=32,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=16,
        # loss metric to optimize
        loss=pyforecast.QuantileLoss(), # allows estimated error bars on predictions
        # logging frequency
        log_interval=2,
        # optimizer parameters
        learning_rate=0.03,
        reduce_on_plateau_patience=4
    )
    return mlp

def get_constant_model(training):
    return pyforecast.Baseline.from_dataset(training)

def get_trainer(model, train_dataloader, val_dataloader, checkpoint_path, checkpoint_filename):
    # create PyTorch Lighning Trainer with early stopping
    early_stop_callback = plcb.EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    checkpoint_callback = plcb.ModelCheckpoint(dirpath=checkpoint_path, filename=checkpoint_filename, monitor='valloss')
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, checkpoint_callback]
    )

    # find the optimal learning rate
    res = trainer.lr_find(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, early_stop_threshold=1000.0, max_lr=0.3,
    )
    return trainer

def train(trainer, model, train_dataloader, val_dataloader):
    # fit the model on the data - redefine the model with the correct learning rate if necessary
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
    )

def test(trainer, model, test_dataloader):
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = model.load_from_checkpoint(best_model_path)
    actuals = torch.cat([y[0] for x, y in iter(test_dataloader)]) # TODO confused by this
    predictions = best_model.predict(test_dataloader)
    metric = pyforecast.MAE()
    acc = metric.loss(predictions, actuals)

def main(args):
    training, validation, testset = get_datasets(args.commodity, args.suffix, args.days_ahead, args.seq_len, args.target)
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(training, validation, testset, args.batch_size)

    model = get_model(training)
    trainer = get_trainer(model, train_dataloader, val_dataloader)
    train(trainer, model, train_dataloader, val_dataloader)
    test(trainer, model, test_dataloader)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)