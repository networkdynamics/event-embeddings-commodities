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

    dataset = prediction.CommodityDataset(args.commodity, args.method, args.days_ahead, seq_len)
    df = dataset.df

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    indices = list(range(len(dataset)))
    
    test_dataset = torch.utils.data.Subset(dataset, indices[-test_size:])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=torch.utils.data.SequentialSampler(test_dataset))

    model = prediction.AttnDecoderRNN(dataset.feature_size, args.hidden_size, combine=args.combine)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    model.load_state_dict(torch.load(model_checkpoint_path))

    close_mean = df['close'].mean()
    close_std = df['close'].std()
    df['predicted'] = 0

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

            last_index = indices[-1]
            final_pred = decoder_output.detach().numpy()
            # normalize
            df.at[last_index + args.days_ahead, 'predicted'] = (final_pred * close_std) + close_mean

    df.to_csv(model_checkpoint_path.replace('.pt', '_predictions.csv'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--commodity')
    parser.add_argument('--days-ahead')
    parser.add_argument('--method')
    parser.add_argument('--combine')
    parser.add_argument('--hidden-size')
    parser.add_argument('--model-checkpoint-name')
    args = parser.parse_args()

    main(args)