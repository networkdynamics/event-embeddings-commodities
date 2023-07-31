import argparse
import json
import os

import numpy as np

import nonlincausality as nlc

import forecasting

import optuna


class Optimize:
    def __init__(self, data_train, data_test):
        self.data_train = data_train
        self.data_test = data_test

    def objective(self, trial):

        lag = 5 # happy medium in terms of previous studies showing length of time of causality for news sentiment
        n_lstm_layers = trial.suggest_int('n_lstm_layers', 1, 2)
        n_lstm_neurons = trial.suggest_int('n_lstm_neurons', 1, 128, log=True)
        n_dense_layers = trial.suggest_int('n_dense_layers', 1, 2)
        n_dense_neurons = trial.suggest_int('n_dense_neurons', 1, 128, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01)

        results = nlc.nonlincausalityLSTM(
            x=self.data_train, 
            maxlag=[lag],
            LSTM_layers=n_lstm_layers,
            LSTM_neurons=[n_lstm_neurons for _ in range(n_lstm_layers)], 
            Dense_layers=n_dense_layers, 
            Dense_neurons=[n_dense_neurons for _ in range(n_dense_layers)], 
            x_test=self.data_test, 
            run=1, 
            add_Dropout=True, 
            Dropout_rate=dropout_rate, 
            epochs_num=[100], 
            learning_rate=[learning_rate], 
            batch_size_num=128, 
            verbose=False, 
            plot=True
        )

        return results[lag].best_RSS_XY

def avg_sentiment(vals):
    return sum(vals) / len(vals) if len(vals) > 0 else np.nan

def avg_embedding(embeddings):
    return np.mean(embeddings, axis=0) if len(embeddings) > 0 else np.nan

def get_data(df, get_split, price_type, feature_type):
    df = df[df['price_type'] == price_type]

    df[f'avg_{feature_type}'] = df[feature_type].apply(avg_sentiment if feature_type == 'sentiment' else avg_embedding)
    df[f'avg_{feature_type}'] = df[f'avg_{feature_type}'].fillna(method='ffill')
    df = df.dropna(subset=['price_diff', f'avg_{feature_type}'])

    df_train = get_split(df, 'train')
    df_test = get_split(df, 'val')

    # normalize price diff and avg sentiment
    df_train['price_diff'] = (df_train['price_diff'] - df_train['price_diff'].mean()) / df_train['price_diff'].std()
    df_test['price_diff'] = (df_test['price_diff'] - df_test['price_diff'].mean()) / df_test['price_diff'].std()
    
    if feature_type == 'sentiment':
        df_train['avg_sentiment'] = (df_train['avg_sentiment'] - df_train['avg_sentiment'].mean()) / df_train['avg_sentiment'].std()
        df_test['avg_sentiment'] = (df_test['avg_sentiment'] - df_test['avg_sentiment'].mean()) / df_test['avg_sentiment'].std()

    data_train = df_train[['price_diff', f'avg_{feature_type}']].values
    data_test = df_test[['price_diff', f'avg_{feature_type}']].values

    if feature_type == 'embedding':
        data_train = np.concatenate([data_train[:, :1], np.stack(item[0] for item in data_train[:, 1:])], axis=1).astype(float)
        data_test = np.concatenate([data_test[:, :1], np.stack(item[0] for item in data_test[:, 1:])], axis=1).astype(float)
    
    return data_train, data_test

def optimize_hyperparameters(data_train, data_test):
    optimizer = Optimize(data_train, data_test)
    study = optuna.create_study(direction='minimize')
    study.optimize(optimizer.objective, n_trials=50)
    return study.best_params

def test_granger_causality(data_train, data_test, lags, method, params):
    if method == 'LSTM':
        results = nlc.nonlincausalityLSTM(
            x=data_train, 
            maxlag=lags, 
            LSTM_layers=params['n_lstm_layers'], 
            LSTM_neurons=[params['n_lstm_neurons'] for _ in range(params['n_lstm_layers'])],
            Dense_layers=params['n_dense_layers'], 
            Dense_neurons=[params['n_dense_neurons'] for _ in range(params['n_dense_layers'])],
            x_test=data_test, 
            run=3, 
            add_Dropout=True, 
            Dropout_rate=params['dropout_rate'], 
            epochs_num=[100], 
            learning_rate=[params['learning_rate']], 
            batch_size_num=128, 
            verbose=False, 
            plot=True
        )
    elif method == 'ARIMA':
        results = nlc.nonlincausalityARIMA(
            x=data_train,
            maxlag=lags,
            x_test=data_test
        )

    return {
        lag: {
            'best_RSS_X': round(result.best_RSS_X, 4), 
            'best_RSS_XY': round(result.best_RSS_XY, 4), 
            'p_value': round(result.p_value, 4)
        } for lag, result in results.items()}

def main(args):
    suffix = args.suffix

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

    feature_type = 'sentiment' if suffix == 'sentiment' else 'embedding'

    # optimize hyperparameters
    if args.method == 'LSTM':
        commodity = 'crude_oil'
        df, article_embeddings, get_split, article_index_cols = forecasting.get_dataframes(commodity, suffix, splits=[0.7, 0.3])
        data_train, data_test = get_data(df, get_split, 'close', feature_type)

        params = optimize_hyperparameters(data_train, data_test)
    else:
        params = None

    for commodity in commodities:
        df, article_embeddings, get_split, article_index_cols = forecasting.get_dataframes(commodity, suffix, splits=[0.7, 0.3])

        lags = [1, 3, 5, 10, 30]
        method = args.method

        price_types = df['price_type'].unique()

        all_results = {}
        for price_type in price_types:
            data_train, data_test = get_data(df, get_split, price_type, feature_type)
            result = test_granger_causality(data_train, data_test, lags, method, params)
            all_results[price_type] = result

        this_dir_path = os.path.dirname(os.path.abspath(__file__))
        results_path = os.path.join(this_dir_path, '..', '..', 'data', 'results', 'commodity')
        with open(os.path.join(results_path, f'{commodity}_{suffix}_{method}_granger.json'), 'w') as f:
            json.dump(all_results, f, indent=4)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)
    args = parser.parse_args()

    main(args)