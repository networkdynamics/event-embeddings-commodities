import argparse
import os

from matplotlib import pyplot as plt
import torch

import prediction

def main(args):

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(this_dir_path, '..', '..', 'checkpoints', 'commodity')
    model_checkpoint_path = os.path.join(checkpoint_path, args.commodity, str(args.days_ahead), args.method, args.model_checkpoint_name)

    batch_size = 1
    seq_len = 50

    method_suffixes = {
        'news2vec': '0521_news2vec_embeds',
        'sentiment': 'sentiment'
    }

    dataset = prediction.CommodityDataset(args.commodity, method_suffixes[args.method], args.days_ahead, seq_len)
    df = dataset.df

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    indices = list(range(len(dataset)))
    
    test_dataset = torch.utils.data.Subset(dataset, indices[-test_size:])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=torch.utils.data.SequentialSampler(test_dataset))

    model = prediction.AttnDecoderRNN(dataset.feature_size, args.hidden_size, combine=args.combine)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))

    df['predicted'] = None

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
            final_pred = decoder_output.detach().numpy()[0]
            df.at[last_index, 'predicted'] = final_pred

    
    if 'diff' in args.model_checkpoint_name:
        df['predicted'] = df['predicted'] + df['norm_close'].shift(1)
    df['predicted'] = df['predicted'].shift(args.days_ahead)
    df['predicted'] = (df['predicted'] * df['close'].std()) + df['close'].mean()

    df.to_csv(model_checkpoint_path.replace('.pt', '_predictions.csv'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--commodity')
    parser.add_argument('--days-ahead', type=int)
    parser.add_argument('--method')
    parser.add_argument('--combine')
    parser.add_argument('--hidden-size', type=int)
    parser.add_argument('--model-checkpoint-name')
    args = parser.parse_args()

    main(args)