import argparse
import datetime
import numpy as np
import os
import sqlite3

# imports for training
from sklearn import preprocessing as skprepro
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plcb
import pandas as pd
import pytorch_forecasting as pyforecast
import torch

class PretrainedMultiEmbeddings(pyforecast.models.nn.MultiEmbedding):
    def __init__(self, *args, **kwargs):
        self.embedding_weights = kwargs.pop('embedding_weights')
        super().__init__(*args, **kwargs)

    def init_embeddings(self):
        self.embeddings = torch.nn.ModuleDict()
        for name in self.embedding_sizes.keys():
            if name not in self.embedding_weights:
                embedding_size = self.embedding_sizes[name][1]
                if self.max_embedding_size is not None:
                    embedding_size = min(embedding_size, self.max_embedding_size)
                # convert to list to become mutable
                self.embedding_sizes[name] = list(self.embedding_sizes[name])
                self.embedding_sizes[name][1] = embedding_size
                
            if name in self.embedding_paddings:
                padding_idx = 0
            else:
                padding_idx = None

            if name in self.categorical_groups:  # embedding bag if related embeddings
                if name in self.embedding_weights:
                    self.embeddings[name] = pyforecast.models.nn.embeddings.TimeDistributedEmbeddingBag.from_pretrained(
                        torch.tensor(self.embedding_weights[name], dtype=torch.float),
                        padding_idx=padding_idx
                    )
                    self.embeddings[name].weight.requires_grad = False
                else:
                    self.embeddings[name] = pyforecast.models.nn.embeddings.TimeDistributedEmbeddingBag(
                        self.embedding_sizes[name][0], embedding_size, mode="sum", batch_first=True
                    )

            else:
                if name in self.embedding_weights:
                    self.embeddings[name] = torch.nn.Embedding.from_pretrained(
                        torch.tensor(self.embedding_weights[name], dtype=torch.float), 
                        padding_idx=padding_idx,
                    )
                    self.embeddings[name].weight.requires_grad = False
                else:
                    self.embeddings[name] = torch.nn.Embedding(
                        self.embedding_sizes[name][0],
                        embedding_size,
                        padding_idx=padding_idx,
                    )


class NewsEncoder(pyforecast.NaNLabelEncoder):
    def fit(self, y: pd.Series, overwrite: bool = False):
        """
        Fit transformer

        Args:
            y (pd.Series): input data to fit on
            overwrite (bool): if to overwrite current mappings or if to add to it.

        Returns:
            NaNLabelEncoder: self
        """
        if not overwrite and hasattr(self, "classes_"):
            offset = len(self.classes_)
        else:
            offset = 0
            self.classes_ = {}

        # determine new classes
        if self.add_nan:
            if self.is_numeric(y):
                nan = np.nan
            else:
                nan = "nan"
            self.classes_[nan] = 0

        for val in np.unique(y):
            if val not in self.classes_:
                self.classes_[val] = int(val) + 1

        self.classes_vector_ = np.array(list(self.classes_.keys()))
        return self

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
        article_df = article_df.reset_index()
        article_df['index'] = article_df['index'].astype(str)
        article_df = article_df.rename(columns={'index': 'article_index'})
        article_df['publish_date'] = pd.to_datetime(article_df['publish_date'])

        if feature == 'embedding':
            article_df['embedding'] = article_df['embedding'].str.strip('[]').replace('\n', '').apply(lambda x: np.fromstring(x, sep=' '))
            feature_size = len(df['embedding'].iloc[0])

        elif feature == 'sentiment':
            article_df['sentiment'] = article_df['sentiment'].apply(lambda x: np.array([x]))
            feature_size = 1

        article_embeddings = np.stack(article_df[feature].values)
        article_embeddings = np.concatenate((np.zeros(article_embeddings[0:1,:].shape), article_embeddings), axis=0)
    
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
                article_index,
                title,
                {feature}
            from
                articles join price on
                publish_date > last_date and publish_date <= date
            '''
        df = pd.read_sql_query(qry, conn)
    else:
        feature = None
        feature_size = 1
        df = price_df

    df = df.groupby(['date', 'open', 'close']).agg(list).reset_index()
    df['date'] = pd.to_datetime(df['date'])

    if suffix != 'constant':
        # perform padding
        max_articles = df[df[feature] != 0][feature].apply(len).max()
        article_index_cols = [f"article_{idx}" for idx in range(max_articles)]
        article_index_df = pd.DataFrame(df['article_index'].tolist(), index=df.index, columns=article_index_cols, dtype=str).fillna("nan")
        df = pd.concat([df, article_index_df], axis=1)

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
    month_col = 'month'
    df[month_col] = df['date'].dt.month.astype(str)
    article_index_col = 'article_index'
    df[group_col] = 'future'
    categorical_encoders = {'article_index': NewsEncoder(add_nan=True).fit(df[article_index_cols].to_numpy().reshape(-1))}
    training = pyforecast.TimeSeriesDataSet(
        df[df[time_idx] < train_size],
        time_idx=time_idx,  # column name of time of observation
        target=future_col,  # column name of target to predict
        group_ids=[group_col],  # column name(s) for timeseries IDs
        max_encoder_length=seq_len,  # how much history to use
        max_prediction_length=days_ahead,  # how far to predict into future
        # covariates known and unknown in the future to inform prediction
        time_varying_known_categoricals=[month_col], # from the tutorial
        time_varying_unknown_reals=[future_col],
        time_varying_unknown_categoricals=[article_index_col],
        variable_groups={article_index_col: article_index_cols},
        categorical_encoders=categorical_encoders
    )

    # create validation dataset using the same normalization techniques as for the training dataset
    validation = pyforecast.TimeSeriesDataSet.from_dataset(training, df[df[time_idx] < train_size + val_size], min_prediction_idx=training.index.time.max() + 1, stop_randomization=True)
    testset = pyforecast.TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=validation.index.time.max() + 1, stop_randomization=True)
    return training, validation, testset, article_embeddings

def get_dataloaders(training, validation, testset, batch_size):
    # convert datasets to dataloaders for training
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=4)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=4)
    test_dataloader = testset.to_dataloader(train=False, batch_size=batch_size, num_workers=4)
    return train_dataloader, val_dataloader, test_dataloader

def get_model(training, article_embeddings):
    # define network to train - the architecture is mostly inferred from the dataset, so that only a few hyperparameters have to be set by the user
    model = pyforecast.NHiTS.from_dataset(
        # dataset
        training,
        # architecture hyperparameters
        hidden_size=256,
        # n_hidden_layers=1,
        dropout=0.1,
        # loss metric to optimize
        loss=pyforecast.QuantileLoss(), # allows estimated error bars on predictions
        # logging frequency
        log_interval=0,
        # optimizer parameters
        learning_rate=0.001,
        optimizer='adam',
        reduce_on_plateau_patience=4,
        embedding_sizes={'article_index': article_embeddings.shape, 'month': 12},
        embedding_paddings=['article_index']
    )
    model.embeddings = PretrainedMultiEmbeddings(
        embedding_sizes=model.hparams.embedding_sizes,
        categorical_groups=model.hparams.categorical_groups,
        embedding_paddings=model.hparams.embedding_paddings,
        x_categoricals=model.hparams.x_categoricals,
        embedding_weights={'article_index': article_embeddings}
    )
    return model

def get_constant_model(training):
    return pyforecast.Baseline.from_dataset(training)

def get_trainer(model, train_dataloader, val_dataloader, checkpoint_path, checkpoint_filename):
    # create PyTorch Lighning Trainer with early stopping
    early_stop_callback = plcb.EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    checkpoint_callback = plcb.ModelCheckpoint(dirpath=checkpoint_path, filename=checkpoint_filename, monitor='val_loss')
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, checkpoint_callback]
    )

    # find the optimal learning rate
    # res = trainer.tuner.lr_find(
    #     model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, early_stop_threshold=1000.0, max_lr=0.3,
    # )
    return trainer

def train(trainer, model, train_dataloader, val_dataloader):
    # fit the model on the data - redefine the model with the correct learning rate if necessary
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
    )

def test(trainer, model, test_dataloader):
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = model.load_from_checkpoint(best_model_path)
    actuals = torch.cat([y[0] for x, y in iter(test_dataloader)])
    predictions = best_model.predict(test_dataloader)
    metric = pyforecast.MAE()
    acc = metric(predictions, actuals)
    return float(acc)

def main(args):
    training, validation, testset, article_embeddings = get_datasets(args.commodity, args.suffix, args.days_ahead, args.seq_len, args.target)
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(training, validation, testset, args.batch_size)

    model = get_model(training, article_embeddings)
    trainer = get_trainer(model, train_dataloader, val_dataloader)
    train(trainer, model, train_dataloader, val_dataloader)
    test(trainer, model, test_dataloader)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)