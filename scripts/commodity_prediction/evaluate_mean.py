import argparse
import json
import os
import pickle
import statistics

import torch

import forecasting

def get_accuracies(commodity, days_ahead, model_checkpoints_dir, suffix, batch_size, seq_len):
    file_names = os.listdir(model_checkpoints_dir)

    train_data, val_data, test_data, article_embeddings = forecasting.get_datasets(commodity, suffix, days_ahead, seq_len)
    train_dataloader, val_dataloader, test_dataloader = forecasting.get_dataloaders(train_data, val_data, test_data, batch_size)

    if suffix == 'constant':
        model = forecasting.get_constant_model(train_data)
    else:
        model_creator = forecasting.ModelCreator(article_embeddings)
        model = model_creator.get_model(train_data)

    accuracies = []
    for file_name in file_names:
        if not file_name.endswith('.ckpt'):
            continue

        try:
            accuracy = forecasting.test(model, test_dataloader, best_model_path=os.path.join(model_checkpoints_dir, file_name))
            accuracies.append(accuracy)
        except Exception as e:
            pass

    return accuracies

def get_best_hidden_and_accuracies(hidden_size_accuracies):

    best_hidden_size = None
    best_hidden_size_accuracy = 9999
    for hidden_size, accuracies in hidden_size_accuracies.items():
        if min(accuracies) < best_hidden_size_accuracy:
            best_hidden_size = hidden_size
            best_hidden_size_accuracy = min(accuracies)

    if best_hidden_size:
        accuracies = hidden_size_accuracies[best_hidden_size]
    else:
        accuracies = []

    return best_hidden_size, accuracies

def get_score(commodity, days_ahead, num_runs, checkpoint_path, suffix, batch_size, seq_len, target, metric, combine, device, ignore_prev_accuracies=False):
    print(f"Testing {commodity}")

    model_checkpoints_dir = os.path.join(checkpoint_path, commodity, str(days_ahead), args.method)

    if not os.path.exists(model_checkpoints_dir):
        os.makedirs(model_checkpoints_dir)

    hidden_size_accuracies = get_accuracies(commodity, days_ahead, model_checkpoints_dir, suffix, batch_size, seq_len, target, metric)
    best_hidden_size, accuracies = get_best_hidden_and_accuracies(hidden_size_accuracies)

    if ignore_prev_accuracies:
        accuracies = []

    score = do_runs(commodity, days_ahead, model_checkpoints_dir, num_runs, suffix, batch_size, seq_len, target, metric, combine, device, best_hidden_size, accuracies)
    return score

def do_runs(commodity, days_ahead, model_checkpoints_dir, num_runs, suffix, batch_size, seq_len, best_params, accuracies, all):
    this_num_runs = num_runs
    this_num_runs -= len(accuracies)

    if this_num_runs > 0:
        train_data, val_data, test_data, article_embeddings = forecasting.get_datasets(commodity, suffix, days_ahead, seq_len)
        train_dataloader, val_dataloader, test_dataloader = forecasting.get_dataloaders(train_data, val_data, test_data, batch_size)

    for _ in range(this_num_runs):
        if not os.path.exists(os.path.dirname(model_checkpoints_dir)):
            os.makedirs(os.path.dirname(model_checkpoints_dir))

        if suffix == 'constant':
            model = forecasting.get_constant_model(train_data)
            trainer_params = {}
        else:
            creator = forecasting.ModelCreator(article_embeddings)
            trainer_args = ['gradient_clip_val']
            model_params = {k: v for k, v in best_params.items() if k not in trainer_args}
            trainer_params = {k: v for k, v in best_params.items() if k in trainer_args}
            model = creator.get_model(train_data, **model_params)

        trainer = forecasting.get_trainer(model_checkpoints_dir, model=model, **trainer_params)

        if suffix != 'constant' and not all:
            forecasting.train(trainer, model, train_dataloader, val_dataloader)

        test_score = forecasting.test(model, test_dataloader, best_model_path=trainer.checkpoint_callback.best_model_path)

        accuracies.append(test_score)

    mean_accuracy = round(statistics.mean(accuracies), 4)
    stddev_accuracy = round(statistics.stdev(accuracies), 2)
    return f'{mean_accuracy}+-{stddev_accuracy}'

def main(args):

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    root_dir_path = os.path.join(this_dir_path, '..', '..')
    checkpoint_path = os.path.join(root_dir_path, 'checkpoints', 'commodity')

    batch_size = 16
    seq_len = 50
    num_runs = 5
    all_commodities = False
    ignore_prev_accuracies = False

    if args.method == 'sentiment':
        suffix = 'sentiment'
    elif args.method == 'news2vec':
        suffix = '0521_news2vec_embeds'
    elif args.method == 'lm_embed':
        suffix = 'lm_both_embed'
    elif args.method =='lm_32_embed':
        suffix = 'lm_32_embed'
    elif args.method =='lm_context_embed':
        suffix = 'lm_context_embed'
    elif args.method =='lm_graph_embed':
        suffix = 'lm_graph_embed'
    elif args.method == 'constant':
        suffix = 'constant'
    elif args.method == 'no_feature':
        suffix = 'no_feature'


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    scores = {}

    if args.variable == 'commodities':

        days_ahead = 1
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

        if suffix != 'constant':
            commodity = 'crude_oil'
            model_checkpoints_dir = os.path.join(checkpoint_path, commodity, str(days_ahead), args.method)
            if not os.path.exists(os.path.join(model_checkpoints_dir, 'tft_study.pkl')):
                train_data, val_data, test_data, article_embeddings = forecasting.get_datasets(commodity, suffix, days_ahead, seq_len)
                train_dataloader, val_dataloader, test_dataloader = forecasting.get_dataloaders(train_data, val_data, test_data, batch_size)
                study = forecasting.optimize_hyperparameters(model_checkpoints_dir, train_dataloader, val_dataloader, article_embeddings)
                best_params = study.best_trial.params
            else:
                with open(os.path.join(model_checkpoints_dir, 'tft_study.pkl'), 'rb') as f:
                    study = pickle.load(f)
                best_params = study.best_trial.params
        else:
            best_params = None

        for commodity in commodities:
            print(f"Testing {commodity}") 

            model_checkpoints_dir = os.path.join(checkpoint_path, commodity, str(days_ahead), args.method)
            if not os.path.exists(model_checkpoints_dir):
                os.makedirs(model_checkpoints_dir)

            if ignore_prev_accuracies:
                accuracies = []
            else:
                accuracies = get_accuracies(commodity, days_ahead, model_checkpoints_dir, suffix, batch_size, seq_len)

            score = do_runs(commodity, days_ahead, model_checkpoints_dir, num_runs, suffix, batch_size, seq_len, best_params, accuracies, all=False)
            scores[commodity] = score

    elif args.variable == 'days_ahead':

        commodity = 'crude_oil'
        day_intervals = [1, 3, 5, 10, 30]

        if suffix != 'constant':
            model_checkpoints_dir = os.path.join(checkpoint_path, commodity, str(1), args.method)
            with open(os.path.join(model_checkpoints_dir, 'tft_study.pkl'), 'rb') as f:
                study = pickle.load(f)
            best_params = study.best_trial.params
        else:
            best_params = None

        for days_ahead in day_intervals:
            print(f"Testing {days_ahead}") 

            model_checkpoints_dir = os.path.join(checkpoint_path, commodity, str(days_ahead), args.method)
            if not os.path.exists(model_checkpoints_dir):
                os.makedirs(model_checkpoints_dir)

            if ignore_prev_accuracies:
                accuracies = []
            else:
                accuracies = get_accuracies(commodity, days_ahead, model_checkpoints_dir, suffix, batch_size, seq_len)

            score = do_runs(commodity, days_ahead, model_checkpoints_dir, num_runs, suffix, batch_size, seq_len, best_params, accuracies, all=False)
            scores[days_ahead] = score

    print(scores)

    if not os.path.exists(os.path.join(root_dir_path, 'data', 'results', 'commodity')):
        os.mkdir(os.path.join(root_dir_path, 'data', 'results', 'commodity'))

    with open(os.path.join(root_dir_path, 'data', 'results', 'commodity', f'{args.method}_{args.variable}.json'), 'w') as f:
        json.dump(scores, f, indent=4)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--method')
    parser.add_argument('--variable')
    args = parser.parse_args()

    main(args)