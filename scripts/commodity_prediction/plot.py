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

    model = prediction.AttnDecoderRNN(feature_size, args.hidden_size, combine=args.combine)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    test_score = prediction.test(model, test_data, device, model_checkpoint_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--commodity')
    parser.add_argument('--days-ahead')
    parser.add_argument('--method')
    parser.add_argument('--combine')
    parser.add_argument('--hidden-size')
    args = parser.parse_args()

    main(args)