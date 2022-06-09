import argparse
import os

from matplotlib import pyplot as plt
import torch

import prediction

def main(args):

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(this_dir_path, '..', '..', 'checkpoints', 'commodity')
    model_checkpoint_path = os.path.join(checkpoint_path, args.commodity, str(args.days_ahead), args.method, args.model_checkpoint_name)

    

    # plot
    fig, ax = plt.subplots()
    ax.plot(df['date'], df['close'], label='Close')
    ax.plot(df['date'], df['predicted'], label='Predicted Close')

    plt.legend()
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--commodity')
    parser.add_argument('--days-ahead')
    parser.add_argument('--method')
    parser.add_argument('--model-checkpoint-name')
    args = parser.parse_args()

    main(args)