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

    batch_size = 16
    days_ahead = 30
    seq_len = 50
    resume = False

    if args.type == 'sentiment':
        suffix = 'sentiment'
        hidden_size = 5
        combine = 'avg'

    commodity_scores = {}

    for commodity in COMMODITIES:
        model_checkpoint_path = os.path.join(checkpoint)
        train_data, val_data, test_data, feature_size = prediction.load_data(commodity, suffix, batch_size, days_ahead, seq_len)

        if not os.path.exists(os.path.dirname(mocheckpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))

        model = prediction.AttnDecoderRNN(feature_size, hidden_size, combine=combine)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        prediction.train(model, train_data, val_data, device, checkpoint_path, resume, days_ahead, seq_len)
        test_score = prediction.test(model, test_data, device, checkpoint_path, days_ahead, seq_len)
        commodity_scores[commodity] = test_score

    print(commodity_scores)

if __name__ == '__main__':
    main()