import argparse
import datetime
import functools
import numpy as np
import os
import pickle
import sqlite3

# imports for training
from sklearn import preprocessing as skprepro
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plcb
import pandas as pd
import pytorch_forecasting as pyforecast
from pytorch_forecasting.models.nhits.tuning import optimize_hyperparameters as nhits_optimize_hyperparameters
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters as tft_optimize_hyperparameters
from pytorch_forecasting.utils import get_embedding_size
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
    def fit(self, y: pd.Series, embeddings, overwrite: bool = False):
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

        for val in range(1, embeddings.shape[0]):
            self.classes_[str(val - 1)] = val

        for val in np.unique(y):
            if val == 'nan':
                continue
            assert self.classes_[val] == int(val) + 1

        self.classes_vector_ = np.array(list(self.classes_.keys()))
        return self

def str_to_numpy(str_array):
    return np.fromstring(str_array[1:-1], sep=' ')

def get_commodity_df(commodity, suffix, article_index_pad=0):
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    commodity_dir = os.path.join(this_dir_path, '..', '..', 'data', 'commodity_data')

    commodity_price_path = os.path.join(commodity_dir, f"{commodity}.csv")
    price_df = pd.read_csv(commodity_price_path)
    price_df = price_df.rename(columns={col: col.lower() for col in price_df.columns})
    price_df = price_df[['date', 'open', 'close', 'low', 'high', 'volume']]
    price_df = price_df.dropna()
    price_df['date'] = pd.to_datetime(price_df['date'])
    price_df = price_df.sort_values('date')
    price_df['last_date'] = price_df['date'].shift(1)
    price_df.at[0, 'last_date'] = pd.Timestamp(price_df.loc[0, 'date'].date() - datetime.timedelta(days=1))

    if suffix not in ['constant', 'no_feature']:
        commodity_article_path = os.path.join(commodity_dir, f"{commodity}_{suffix}.csv")
        article_df = pd.read_csv(commodity_article_path)#, dtype={'publish_date': str, 'title': str, 'embedding': str, 'embed': str, 'sentiment': float})

        article_df = article_df.dropna()

        article_df = article_df.rename(columns={'embed': 'embedding'})
        feature = [feat_type for feat_type in ['sentiment', 'embedding'] if feat_type in article_df.columns][0]

        article_df = article_df[['title', 'publish_date', feature]]
        article_df = article_df.reset_index()
        # add article index pad
        article_df['index'] = article_df['index'].astype(int) + article_index_pad
        article_df['index'] = article_df['index'].astype(str)
        article_df = article_df.rename(columns={'index': 'article_index'})
        article_df['publish_date'] = pd.to_datetime(article_df['publish_date'])

        # trim price df to only from when we have articles
        price_df = price_df[price_df['last_date'] >= article_df['publish_date'].min()]

        if feature == 'embedding':
            article_df['embedding_np'] = article_df['embedding'].str.strip('[]').replace('\n', '').apply(lambda x: np.fromstring(x, sep=' '))

        elif feature == 'sentiment':
            article_df['sentiment_np'] = article_df['sentiment'].apply(lambda x: np.array([x]))

        article_embeddings = np.stack(article_df[f'{feature}_np'].values)
        # stack the null feature at index 0
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
                low,
                high,
                volume,
                article_index,
                title,
                {feature}
            from price 
                left join articles on
                publish_date > last_date and publish_date <= date
            '''
        df = pd.read_sql_query(qry, conn)
    else:
        feature = None
        article_embeddings = None
        df = price_df

    def list_without_nan_or_none(vals):
        return list(vals.dropna())

    df = df.groupby(['date', 'open', 'close', 'low', 'high', 'volume']).agg(list_without_nan_or_none).reset_index()
    df['date'] = pd.to_datetime(df['date'])

    if feature == 'embedding':
        df['embedding'] = df['embedding'].apply(lambda x: [str_to_numpy(str_array) for str_array in x])

    if suffix not in ['constant', 'no_feature']:
        # perform padding
        max_articles = df[df[feature] != 0][feature].apply(len).max()
        article_index_cols = [f"article_{idx}" for idx in range(max_articles)]
        article_index_df = pd.DataFrame(df['article_index'].tolist(), index=df.index, columns=article_index_cols, dtype=str).fillna("nan")
        df = pd.concat([df, article_index_df], axis=1)
        id_vars = ['time_index', 'date', 'volume', 'article_index', 'title', feature] + article_index_cols
    else:
        article_index_cols = []
        id_vars = ['time_index', 'date', 'volume']

    df = df.reset_index()
    df = df.rename(columns={'index': 'time_index'})

    price_type_col = 'price_type'
    price_col = 'price'
    # melt so we can use all time series
    df = df.melt(
        id_vars=id_vars,
        value_vars=['open', 'close', 'low', 'high'],
        var_name=price_type_col,
        value_name=price_col
    )

    # get price diffs
    df = df.sort_values(['price_type', 'time_index'])
    df['price_diff'] = df.groupby(price_type_col)[price_col].diff()

    return df, article_index_cols, article_embeddings

def get_dataframes(commodity, suffix, splits=[0.7, 0.15, 0.15]):
    time_idx = 'time_index'

    if commodity == 'all':
        commodity_type_col = 'commodity_type'

        df = pd.DataFrame()
        article_index_cols = []
        article_embeddings = None
        commodities = [
            'crude_oil',
            'brent_crude_oil',
            'natural_gas',
            'rbob_gasoline',
            'copper',
            'palladium',
            'platinum',
            'gold',
            'silver',
            'corn',
            'cotton',
            'soybean',
            'sugar',
            'wheat'
        ]
        commodities.reverse()
        article_index_pad = 0
        com_lens = {}
        for com in commodities:
            com_df, com_article_index_cols, com_article_embeddings = get_commodity_df(com, suffix, article_index_pad=article_index_pad)
            com_df[commodity_type_col] = com
            com_lens[com] = com_df[time_idx].max()
            df = pd.concat([df, com_df])
            article_index_cols = list(set(article_index_cols).union(com_article_index_cols))
            article_index_pad += com_article_embeddings.shape[0]
            if article_embeddings is not None:
                article_embeddings = np.concatenate([article_embeddings, com_article_embeddings])
            else:
                article_embeddings = com_article_embeddings

        df[article_index_cols] = df[article_index_cols].fillna("nan")
        df = df.reset_index(drop=True)

        # make train split equal across commodities
        def get_split(df, split):
            com_splits = {com: (int(splits[0] * com_len), int(splits[1] * com_len)) for com, com_len in com_lens.items()}
            if split == 'train':
                filter = functools.reduce(
                    np.logical_or, 
                    [(df[time_idx] < com_split[0]) & (df[commodity_type_col] == com) for com, com_split in com_splits.items()]
                )
                return df[filter]
            elif split == 'val':
                filter = functools.reduce(
                    np.logical_or, 
                    [(df[time_idx] >= com_split[0]) & (df[time_idx] < com_split[0] + com_split[1]) & (df[commodity_type_col] == com) for com, com_split in com_splits.items()]
                )
                return df[filter]
            elif split == 'test':
                filter = functools.reduce(
                    np.logical_or, 
                    [(df[time_idx] >= com_split[0] + com_split[1]) & (df[commodity_type_col] == com) for com, com_split in com_splits.items()]
                )
                return df[filter]
    else:
        df, article_index_cols, article_embeddings = get_commodity_df(commodity, suffix)

        dataset_size = df[time_idx].max()
        train_size = int(splits[0] * dataset_size)
        val_size = int(splits[1] * dataset_size)

        def get_split(df, split):
            if split == 'train':
                return df[df[time_idx] < train_size]
            elif split == 'val':
                return df[(df[time_idx] >= train_size) & (df[time_idx] < train_size + val_size)]
            elif split == 'test':
                return df[df[time_idx] >= train_size + val_size]

    # load data: this is pandas dataframe with at least a column for
    # * the target (what you want to predict)
    # * the timeseries ID (which should be a unique string to identify each timeseries)
    # * the time of the observation (which should be a monotonically increasing integer)
    
    month_col = 'month'
    df[month_col] = df['date'].dt.month.astype(str)

    return df, article_embeddings, get_split, article_index_cols

def get_datasets(commodity, suffix, days_ahead, seq_len, splits=[0.7, 0.15, 0.15]):
    df, article_embeddings, get_split, article_index_cols = get_dataframes(commodity, suffix, splits=[0.7, 0.15, 0.15])
    return _get_datasets(commodity, suffix, days_ahead, seq_len, df, article_embeddings, get_split, article_index_cols)

def _get_datasets(commodity, suffix, days_ahead, seq_len, df, article_embeddings, get_split, article_index_cols):
    time_idx = 'time_index'
    price_type_col = 'price_type'
    month_col = 'month'
    price_col = 'price_diff'
    volume_col = 'volume'
    article_index_col = 'article_index'

    if commodity == 'all':
        commodity_type_col = 'commodity_type'
        static_categoricals = [commodity_type_col, price_type_col]
    else:
        static_categoricals = [price_type_col]

    if price_col == 'price_diff':
        df = df.dropna(subset=[price_col])

    train_split = get_split(df, 'train')
    val_split = get_split(df, 'val')
    test_split = get_split(df, 'test')

    if suffix not in ['constant', 'no_feature']:
        time_varying_unknown_categoricals = [article_index_col]
        variable_groups = {article_index_col: article_index_cols}
        categorical_encoders = {'article_index': NewsEncoder(add_nan=True).fit(df[article_index_cols].to_numpy().reshape(-1), article_embeddings)}
    else:
        time_varying_unknown_categoricals = []
        variable_groups = {}
        categorical_encoders = {}

    training = pyforecast.TimeSeriesDataSet(
        train_split,
        time_idx=time_idx,  # column name of time of observation
        target=price_col,  # column name of target to predict
        group_ids=static_categoricals,  # column name(s) for timeseries IDs
        max_encoder_length=seq_len,  # how much history to use
        max_prediction_length=days_ahead,  # how far to predict into future
        # covariates known and unknown in the future to inform prediction
        static_categoricals=static_categoricals,
        time_varying_known_reals=[time_idx],
        time_varying_known_categoricals=[month_col],
        time_varying_unknown_reals=[price_col, volume_col],
        time_varying_unknown_categoricals=time_varying_unknown_categoricals,
        variable_groups=variable_groups,
        categorical_encoders=categorical_encoders
    )

    # create validation dataset using the same normalization techniques as for the training dataset
    
    validation = pyforecast.TimeSeriesDataSet.from_dataset(training, val_split, stop_randomization=True)
    testset = pyforecast.TimeSeriesDataSet.from_dataset(training, test_split, stop_randomization=True)
    return training, validation, testset, article_embeddings

def get_dataloaders(training, validation, testset, batch_size):
    # convert datasets to dataloaders for training
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=4)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=4)
    test_dataloader = testset.to_dataloader(train=False, batch_size=batch_size, num_workers=4)
    return train_dataloader, val_dataloader, test_dataloader

def get_nhits_model(training, article_embeddings, hidden_size=256, dropout=0.1, log_interval=0, learning_rate=0.001, loss=pyforecast.QuantileLoss(), reduce_on_plateau_patience=100, **kwargs):
    model = pyforecast.NHiTS.from_dataset(
        # dataset
        training,
        # architecture hyperparameters
        hidden_size=hidden_size,
        # n_hidden_layers=1,
        dropout=dropout,
        # loss metric to optimize
        loss=loss, # allows estimated error bars on predictions
        # logging frequency
        log_interval=log_interval,
        # optimizer parameters
        learning_rate=learning_rate,
        optimizer='adam',
        reduce_on_plateau_patience=reduce_on_plateau_patience,
        embedding_sizes={'article_index': article_embeddings.shape}, #, 'month': 12},
        embedding_paddings=['article_index'],
        **kwargs
    )

    model.embeddings = PretrainedMultiEmbeddings(
        embedding_sizes=model.hparams.embedding_sizes,
        categorical_groups=model.hparams.categorical_groups,
        embedding_paddings=model.hparams.embedding_paddings,
        x_categoricals=model.hparams.x_categoricals,
        embedding_weights={'article_index': article_embeddings}
    )

    return model

def get_embedding_sizes(training, article_embeddings):
    embedding_sizes = {}
    if article_embeddings is not None:
        embedding_sizes['article_index'] = article_embeddings.shape
        embedding_paddings = ['article_index']
    else:
        embedding_paddings = []

    if 'price_type' in training.categoricals:
        embedding_sizes['price_type'] = (4, get_embedding_size(4))

    if 'month' in training.categoricals:
        embedding_sizes['month'] = (12, get_embedding_size(12))

    if 'commodity_type' in training.categoricals:
        embedding_sizes['commodity_type'] = (14, get_embedding_size(14))

    return embedding_sizes, embedding_paddings

def get_tft_model(training, article_embeddings, hidden_size=32, dropout=0.1, log_interval=0, learning_rate=0.001, loss=pyforecast.QuantileLoss(), reduce_on_plateau_patience=100, **kwargs):
    embedding_sizes, embedding_paddings = get_embedding_sizes(training, article_embeddings)

    model = pyforecast.TemporalFusionTransformer.from_dataset(
        # dataset
        training,
        # architecture hyperparameters
        hidden_size=hidden_size,
        # n_hidden_layers=1,
        dropout=dropout,
        # loss metric to optimize
        loss=loss, # allows estimated error bars on predictions
        # logging frequency
        log_interval=log_interval,
        # optimizer parameters
        learning_rate=learning_rate,
        optimizer='adam',
        reduce_on_plateau_patience=reduce_on_plateau_patience,
        embedding_sizes=embedding_sizes,
        embedding_paddings=embedding_paddings,
        **kwargs
    )

    if article_embeddings is not None:
        embedding_weights = {'article_index': article_embeddings}
        model.input_embeddings = PretrainedMultiEmbeddings(
            embedding_sizes=model.hparams.embedding_sizes,
            categorical_groups=model.hparams.categorical_groups,
            embedding_paddings=model.hparams.embedding_paddings,
            x_categoricals=model.hparams.x_categoricals,
            embedding_weights=embedding_weights
        )

    return model

class ModelCreator:
    def __init__(self, article_embeddings):
        self.article_embeddings = article_embeddings

    def get_model(self, training, hidden_size=32, dropout=0.1, log_interval=0, learning_rate=0.001, loss=pyforecast.QuantileLoss(), reduce_on_plateau_patience=100, **kwargs):
        # define network to train - the architecture is mostly inferred from the dataset, so that only a few hyperparameters have to be set by the user
        model = get_tft_model(training, self.article_embeddings, 
            hidden_size=hidden_size,
            dropout=dropout,
            log_interval=log_interval,
            learning_rate=learning_rate,
            loss=loss,
            reduce_on_plateau_patience=reduce_on_plateau_patience,
            **kwargs
        )
        return model

def get_constant_model(training):
    return pyforecast.Baseline.from_dataset(training)

def get_trainer(checkpoint_path, model=None, checkpoint_filename=None, gradient_clip_val=0.1):
    if not checkpoint_filename:
        checkpoint_filename = type(model).__name__.lower() + '-{val_loss:.4f}'
    # create PyTorch Lighning Trainer with early stopping
    early_stop_callback = plcb.EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=10, verbose=False, mode="min")
    checkpoint_callback = plcb.ModelCheckpoint(dirpath=checkpoint_path, filename=checkpoint_filename, monitor='val_loss')
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        gradient_clip_val=gradient_clip_val,
        callbacks=[early_stop_callback, checkpoint_callback]
    )

    return trainer

def train(trainer, model, train_dataloader, val_dataloader):
    # fit the model on the data - redefine the model with the correct learning rate if necessary
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
    )

def test(model, test_dataloader, best_model_path=None):
    if best_model_path:
        best_model = model.load_from_checkpoint(best_model_path)
    else:
        best_model = model
    actuals = torch.cat([y[0] for x, y in iter(test_dataloader)])
    predictions = best_model.predict(test_dataloader)
    metric = pyforecast.MAE()
    acc = metric(predictions, actuals)
    return float(acc)

def optimize_hyperparameters(dir_path, train_dataloader, val_dataloader, article_embeddings):

    study_path = os.path.join(dir_path, "tft_study.pkl")
    if os.path.exists(study_path):
        with open(study_path, 'rb') as f:
            study = pickle.load(f)
    else:
        model_creator = ModelCreator(article_embeddings)
        # create study
        study = tft_optimize_hyperparameters(
            train_dataloader,
            val_dataloader,
            model_creator.get_model,
            model_path=dir_path,
            hidden_size_range=(max(16, article_embeddings.shape[1]), 265),
            n_trials=200,
            max_epochs=50,
            trainer_kwargs=dict(limit_train_batches=30),
            reduce_on_plateau_patience=4,
            use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
        )

        # save study results - also we can resume tuning at a later point in time
        with open(study_path, "wb") as fout:
            pickle.dump(study, fout)

    return study

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