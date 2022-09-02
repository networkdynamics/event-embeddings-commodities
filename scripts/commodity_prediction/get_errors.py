import argparse
import os
import re

from matplotlib import pyplot as plt
import torch

import prediction

def main():

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(this_dir_path, '..', '..', 'checkpoints', 'commodity')

    batch_size = 1
    seq_len = 50

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

    methods = {
        'sentiment': {
            'suffix': 'sentiment',
            'hidden_sizes': [3,4,8],
            'combine': 'avg'
        },
        'news2vec': {
            'suffix': '0521_news2vec_embeds',
            'hidden_sizes': [32, 48, 64],
            'combine': 'attn'
        },
        'lm_embed': {
            'suffix': 'lm_small_embed',
            'hidden_sizes': [32, 48, 64, 16],
            'combine': 'attn'
        }
    }

    days_ahead = 30
    target = 'price'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for commodity in commodities:
        for method in methods:

            dataset = prediction.CommodityDataset(commodity, methods[method]['suffix'], days_ahead, seq_len, target=target)
            df = dataset.df

            train_size = int(0.8 * len(dataset))
            val_size = int(0.1 * len(dataset))
            test_size = len(dataset) - train_size - val_size
            indices = list(range(len(dataset)))
            
            test_dataset = torch.utils.data.Subset(dataset, indices[-test_size:])
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=torch.utils.data.SequentialSampler(test_dataset))


            model_checkpoints_dir = os.path.join(checkpoint_path, commodity, str(days_ahead), method)
            file_names = os.listdir(model_checkpoints_dir)
            accuracies = {}
            for file_name in file_names:
                if not re.match('.*final[_]+model.pt', file_name):
                    continue

                if 'dir' in file_name or 'diff' in file_name:
                    continue

                model_checkpoint_path = os.path.join(model_checkpoints_dir, file_name)

                for hidden_size in methods[method]['hidden_sizes']:
                    model = prediction.AttnDecoderRNN(dataset.feature_size, hidden_size, combine=methods[method]['combine'])
                    model = model.to(device)
                    try:
                        model.load_state_dict(torch.load(model_checkpoint_path))
                        break
                    except RuntimeError:
                        continue
                else:
                    print('No hidden size worked')
                    continue

                model.eval()

                df['predicted'] = None
                if methods[method]['combine'] == 'attn':
                    df['important_title'] = None
                    df['important_title_attention'] = None

                with torch.no_grad():
                    for batch in test_dataloader:
                        indices = batch['index']
                        encoder_outputs = batch['encoder_outputs'].to(device)
                        attention_masks = batch['attention_mask'].to(device)
                        inputs = batch['inputs'].to(device)
                        targets = batch['targets'].to(device)
                        
                        target_length = targets.shape[1]

                        decoder_hidden = model.init_hidden(batch_size, device)

                        for idx in range(target_length):
                            decoder_input = inputs[:, idx]
                            decoder_output, decoder_hidden, decoder_attention = model(
                                decoder_input, decoder_hidden, encoder_outputs[:,idx,:,:], attention_masks[:,idx,:])

                        last_index = int(indices[0][-1].detach().numpy())

                        if methods[method]['combine'] == 'attn':
                            attention = decoder_attention.squeeze(0).squeeze(1)
                            important_title_attention = float(attention.max())
                            important_title_idx = int(attention.argmax())
                            df.at[last_index, 'important_title_attention'] = important_title_attention
                            df.at[last_index, 'important_title'] = df.loc[last_index, 'title'][important_title_idx]

                        final_pred = decoder_output.cpu().detach().numpy()[0]

                        df.at[last_index, 'predicted'] = final_pred

                if 'diff' in file_name:
                    df['predicted'] = df['predicted'] + df['norm_close'].shift(1)

                df['predicted'] = (df['predicted'] * df['close'].std()) + df['close'].mean()
                df['predicted'] = df['predicted'].shift(days_ahead)

                df = df[df['predicted'].notnull()]
            
                df['norm_close'] = (df['close'] - df['close'].mean()) / df['close'].std()
                df['norm_pred'] = (df['predicted'] - df['predicted'].mean()) / df['predicted'].std()

                df['squared_error'] = (df['norm_pred'] - df['norm_close']) ** 2
                mse_accuracy = df['squared_error'].mean()

                accuracies[file_name] = mse_accuracy

            x = 1


if __name__ == '__main__':
    main()