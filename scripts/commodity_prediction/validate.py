import argparse
import os

import torch

import prediction


def main(args):

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(this_dir_path, '..', '..', 'checkpoints', 'commodity')

    batch_size = 32
    seq_len = 50
    resume = False

    if args.method == 'sentiment':
        suffix = 'sentiment'
        hidden_sizes = [3,4,8]
        combine = 'avg'
    elif args.method == 'news2vec':
        suffix = '0521_news2vec_embeds'
        hidden_sizes = [32, 48, 64]
        combine = 'attn'
    elif args.method == 'lm_embed':
        suffix = 'lm_small_embed'
        hidden_sizes = [48, 64]
        combine = 'attn'

    scores = {}

    days_ahead = 30
    commodities = [
        'brent_crude_oil',
        'crude_oil',
        'natural_gas',
        'rbob_gasoline',
        'copper',
        'palladium',
        'platinum',
        'gold',
        'silver',
        'corn',
        'cotton',
        'soybean_meal',
        'soybean_oil',
        'soybean',
        'sugar',
        'wheat'
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for commodity in commodities:
        print(f"Testing {commodity}")
        model_checkpoint_dir = os.path.join(checkpoint_path, commodity, str(days_ahead), args.method)
        best_test_score = 9999

        train_data, val_data, test_data, feature_size = prediction.load_data(commodity, suffix, batch_size, days_ahead, seq_len)

        for model_file_name in os.listdir(model_checkpoint_dir):
            
            if 'final_model' not in model_file_name:
                continue

            specified_test_score = model_file_name[:6]

            model_checkpoint_path = os.path.join(checkpoint_path, commodity, str(days_ahead), args.method, model_file_name)
            
            for hidden_size in hidden_sizes:
                model = prediction.AttnDecoderRNN(feature_size, hidden_size, combine=combine)
                model = model.to(device)
                try:
                    model.load_state_dict(torch.load(model_checkpoint_path))
                    break
                except RuntimeError:
                    continue
            else:
                raise Exception('No hidden size worked')
            
            test_score = prediction.test(model, test_data, device, model_checkpoint_path)
            score_slug = f'{test_score:.4f}'.replace('.', '_')

            if score_slug != specified_test_score:
                os.rename(model_checkpoint_path, model_checkpoint_path.replace(specified_test_score, score_slug))
            else:
                if test_score < best_test_score:
                    best_test_score = test_score

        print(f"Best test score: {best_test_score:.4f}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--method')
    args = parser.parse_args()

    main(args)