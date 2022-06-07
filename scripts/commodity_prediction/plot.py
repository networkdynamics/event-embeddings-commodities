import argparse
import os

import torch

import prediction

def main(args):

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(this_dir_path, '..', '..', 'checkpoints', 'commodity')
    model_checkpoint_path = os.path.join(checkpoint_path, args.commodity, str(args.days_ahead), args.method, 'final_model.pt')

    batch_size = 1
    seq_len = 50

    dataset = prediction.CommodityDataset(args.commodity, args.method, args.days_ahead, seq_len)

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

    with torch.no_grad():
        for batch in test_dataloader:
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--commodity')
    parser.add_argument('--days-ahead')
    parser.add_argument('--method')
    parser.add_argument('--combine')
    parser.add_argument('--hidden-size')
    args = parser.parse_args()

    main(args)