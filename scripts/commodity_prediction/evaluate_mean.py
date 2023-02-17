import argparse
import collections
import os
import re
import statistics

import torch

import forecasting

def get_accuracies(model_checkpoints_dir, target, metric):
    accuracies = []

    file_names = os.listdir(model_checkpoints_dir)

    target_name = '' if target == 'price' else f"_{target}"
    metric_name = '' if metric == 'all' else f"_{metric}"

    hidden_size_accuracies = collections.defaultdict(list)
    for file_name in file_names:
        if not re.match(f'.*final{target_name}{metric_name}_model.pt', file_name):
            continue

        model_checkpoint_path = os.path.join(model_checkpoints_dir, file_name)
        model_state_dict = torch.load(model_checkpoint_path)
        hidden_size = model_state_dict['out.weight'].shape[1]

        acc_regex = '[0-9]{1}_[0-9]{4}'
        acc_match = re.search(acc_regex, file_name)
        if not acc_match:
            continue

        acc_str = acc_match.group(0)
        accuracy = float(acc_str.replace('_', '.'))
        hidden_size_accuracies[hidden_size].append(accuracy)

    return hidden_size_accuracies

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

    hidden_size_accuracies = get_accuracies(model_checkpoints_dir)
    best_hidden_size, accuracies = get_best_hidden_and_accuracies(hidden_size_accuracies)

    if ignore_prev_accuracies:
        accuracies = []

    score = do_runs(commodity, days_ahead, model_checkpoints_dir, num_runs, suffix, batch_size, seq_len, target, metric, combine, device, best_hidden_size, accuracies)
    return score

def do_runs(commodity, days_ahead, model_checkpoints_dir, num_runs, suffix, batch_size, seq_len, combine, best_hidden_size, accuracies):
    this_num_runs = num_runs
    this_num_runs -= len(accuracies)

    if this_num_runs > 0:
        train_data, val_data, test_data, article_embeddings = forecasting.get_datasets(commodity, suffix, days_ahead, seq_len)
        train_dataloader, val_dataloader, test_dataloader = forecasting.get_dataloaders(train_data, val_data, test_data, batch_size)

    for _ in range(this_num_runs):
        model_checkpoint_filename = 'mlp'
        model_checkpoint_path = os.path.join(model_checkpoints_dir, model_checkpoint_filename)
        
        if not os.path.exists(os.path.dirname(model_checkpoint_path)):
            os.makedirs(os.path.dirname(model_checkpoint_path))

        if suffix == 'constant':
            model = forecasting.get_constant_model(train_data)
        else:
            model = forecasting.get_model(train_data, article_embeddings)

        trainer = forecasting.get_trainer(model, train_dataloader, val_dataloader, model_checkpoints_dir, model_checkpoint_filename)

        if suffix != 'constant':
            forecasting.train(trainer, model, train_dataloader, val_dataloader)

        test_score = forecasting.test(trainer, model, test_dataloader)
        test_score = round(test_score, 4)
        score_slug = f'{test_score:.4f}'.replace('.', '_')
        
        if suffix != 'constant':
            os.rename(model_checkpoint_path, model_checkpoint_path.replace('final', f'{score_slug}_final'))

        accuracies.append(test_score)

    mean_accuracy = round(statistics.mean(accuracies), 4)
    stddev_accuracy = round(statistics.stdev(accuracies), 2)
    return f'{mean_accuracy}+-{stddev_accuracy}'

def main(args):

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(this_dir_path, '..', '..', 'checkpoints', 'commodity')

    batch_size = 16
    seq_len = 50
    num_runs = 5
    target = 'price'
    metric = 'last'
    ignore_prev_accuracies = True

    default_hidden_size = None
    if args.method == 'sentiment':
        suffix = 'sentiment'
        combine = 'avg'
        default_hidden_size = 32
    elif args.method == 'news2vec':
        suffix = '0521_news2vec_embeds'
        combine = 'attn'
    elif args.method == 'lm_embed':
        suffix = 'lm_both_embed'
        combine = 'attn'
    elif args.method =='lm_32_embed':
        suffix = 'lm_32_embed'
        combine = 'attn'
    elif args.method =='lm_context_embed':
        suffix = 'lm_context_embed'
        combine = 'attn'
    elif args.method =='lm_graph_embed':
        suffix = 'lm_graph_embed'
        combine = 'attn'
    elif args.method == 'constant':
        suffix = 'constant'
        combine = None


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    scores = {}

    if args.variable == 'commodities':

        days_ahead = 30
        commodities = [
            'soybean',
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
            'sugar',
            'wheat'
        ]

        # if best_hidden_size is None:
        #     commodity = 'crude_oil'
        #     model_checkpoints_dir = os.path.join(checkpoint_path, commodity, str(days_ahead), args.method)

        #     hidden_size_accuracies = get_accuracies(model_checkpoints_dir)
        #     best_hidden_size, _ = get_best_hidden_and_accuracies(hidden_size_accuracies)

        for commodity in commodities:
            print(f"Testing {commodity}") 

            model_checkpoints_dir = os.path.join(checkpoint_path, commodity, str(days_ahead), args.method)
            if not os.path.exists(model_checkpoints_dir):
                os.makedirs(model_checkpoints_dir)

            #hidden_size_accuracies = get_accuracies(model_checkpoints_dir, target, metric)
            #best_hidden_size, accuracies = get_best_hidden_and_accuracies(hidden_size_accuracies)
            #if not best_hidden_size:
            #    best_hidden_size = default_hidden_size

            if ignore_prev_accuracies:
                accuracies = []

            best_hidden_size = default_hidden_size
            score = do_runs(commodity, days_ahead, model_checkpoints_dir, num_runs, suffix, batch_size, seq_len, combine, best_hidden_size, accuracies)
            scores[commodity] = score

    elif args.variable == 'days_ahead':

        commodity = 'crude_oil'
        day_intervals = [
            1,
            10,
            30,
            50,
            100
        ]

        # if best_hidden_size is None:
        #     days_ahead = 30
        #     model_checkpoints_dir = os.path.join(checkpoint_path, commodity, str(days_ahead), args.method)

        #     hidden_size_accuracies = get_accuracies(model_checkpoints_dir)
        #     best_hidden_size, _ = get_best_hidden_and_accuracies(hidden_size_accuracies)

        for days_ahead in day_intervals:
            print(f"Testing {commodity}") 

            model_checkpoints_dir = os.path.join(checkpoint_path, commodity, str(days_ahead), args.method)
            if not os.path.exists(model_checkpoints_dir):
                os.mkdir(model_checkpoints_dir)

            hidden_size_accuracies = get_accuracies(model_checkpoints_dir)
            best_hidden_size, accuracies = get_best_hidden_and_accuracies(hidden_size_accuracies)

            if ignore_prev_accuracies:
                accuracies = []

            score = do_runs(commodity, days_ahead, model_checkpoints_dir, num_runs, suffix, batch_size, seq_len, target, metric, combine, device, best_hidden_size, accuracies)
            scores[days_ahead] = score

    print(scores)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--method')
    parser.add_argument('--variable')
    args = parser.parse_args()

    main(args)