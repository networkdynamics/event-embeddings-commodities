import argparse
import os

import torch

import prediction

COMMODITIES = [
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
    'oat',
    'cotton',
    'soybean_meal',
    'soybean_oil',
    'soybean',
    'sugar',
    'wheat'
]

def main(args):

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(this_dir_path, '..', '..', 'checkpoints', 'commodity')

    batch_size = 32
    days_ahead = 30
    seq_len = 50
    resume = False

    if args.type == 'sentiment':
        suffix = 'sentiment'
        hidden_size = 3
        combine = 'avg'
    elif args.type == 'news2vec':
        suffix = '0521_news2vec_embeds'
        hidden_size = 48
        combine = 'attn'

    commodity_scores = {}

    for commodity in COMMODITIES:
        model_checkpoint_path = os.path.join(checkpoint_path, commodity, str(days_ahead), args.type, 'final_model.pt')
        train_data, val_data, test_data, feature_size = prediction.load_data(commodity, suffix, batch_size, days_ahead, seq_len)

        if not os.path.exists(os.path.dirname(model_checkpoint_path)):
            os.makedirs(os.path.dirname(model_checkpoint_path))

        model = prediction.AttnDecoderRNN(feature_size, hidden_size, combine=combine)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        prediction.train(model, train_data, val_data, device, model_checkpoint_path, resume)
        test_score = prediction.test(model, test_data, device, model_checkpoint_path)
        score_slug = f'{test_score:.4f}'.replace('.', '_')
        os.rename(model_checkpoint_path, model_checkpoint_path.replace('final_model', f'{score_slug}_final_model'))
        commodity_scores[commodity] = test_score

    print(commodity_scores)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--type')
    args = parser.parse_args()

    main(args)