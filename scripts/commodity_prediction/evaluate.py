import argparse
import os

import torch

import prediction


def main(args):

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(this_dir_path, '..', '..', 'checkpoints', 'commodity')

    batch_size = 16
    seq_len = 50
    resume = False
    target = 'price'
    metric = 'last'

    target_name = '' if target == 'price' else f"_{target}"
    metric_name = '' if metric == 'all' else f"_{metric}"

    if args.method == 'sentiment':
        suffix = 'sentiment'
        hidden_size = 64 # 3, 4, 8, 16, 32
        combine = 'avg'
    elif args.method == 'news2vec':
        suffix = '0521_news2vec_embeds'
        hidden_size = 64 #32, 48, 56, 64
        combine = 'attn'
    elif args.method == 'lm_embed':
        suffix = 'lm_small_embed'
        hidden_size = 48 #32, 48, 56, 64
        combine = 'attn'
    elif args.method =='lm_32_embed':
        suffix = 'lm_32_embed'
        hidden_size = 48 # 32, 48, 64, 128
        combine = 'attn'

    scores = {}

    if args.variable == 'commodities':

        days_ahead = 30
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

        for commodity in commodities:
            print(f"Testing {commodity}")
            model_checkpoint_path = os.path.join(checkpoint_path, commodity, str(days_ahead), args.method, f'final{target_name}{metric_name}_model.pt')
            train_data, val_data, test_data, feature_size = prediction.load_data(commodity, suffix, batch_size, days_ahead, seq_len, target)

            if not os.path.exists(os.path.dirname(model_checkpoint_path)):
                os.makedirs(os.path.dirname(model_checkpoint_path))

            model = prediction.AttnDecoderRNN(feature_size, hidden_size, combine=combine)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)

            prediction.train(model, train_data, val_data, device, model_checkpoint_path, resume, days_ahead, target, metric)
            test_score = prediction.test(model, test_data, device, model_checkpoint_path, target, metric)
            score_slug = f'{test_score:.4f}'.replace('.', '_')
            os.rename(model_checkpoint_path, model_checkpoint_path.replace('final', f'{score_slug}_final'))
            scores[commodity] = test_score

    elif args.variable == 'days_ahead':

        commodity = 'crude_oil'
        day_intervals = [
            1,
            10,
            30,
            50,
            100
        ]

        for days_ahead in day_intervals:
            print(f"Testing {commodity}")
            model_checkpoint_path = os.path.join(checkpoint_path, commodity, str(days_ahead), args.method, f'final{target_name}{metric_name}_model.pt')
            train_data, val_data, test_data, feature_size = prediction.load_data(commodity, suffix, batch_size, days_ahead, seq_len)

            if not os.path.exists(os.path.dirname(model_checkpoint_path)):
                os.makedirs(os.path.dirname(model_checkpoint_path))

            model = prediction.AttnDecoderRNN(feature_size, hidden_size, combine=combine)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)

            prediction.train(model, train_data, val_data, device, model_checkpoint_path, resume, days_ahead, target, metric)
            test_score = prediction.test(model, test_data, device, model_checkpoint_path, target, metric)
            score_slug = f'{test_score:.4f}'.replace('.', '_')
            os.rename(model_checkpoint_path, model_checkpoint_path.replace('final', f'{score_slug}_final'))
            scores[days_ahead] = test_score

    print(scores)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--method')
    parser.add_argument('--variable')
    args = parser.parse_args()

    main(args)