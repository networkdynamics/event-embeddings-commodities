import os

import torch
import transformers

import lm_embed

def main():
    large_embed_size = 128
    small_embed_size = 32
    lm_large_model = lm_embed.LanguageModelEmbed(embed_size=large_embed_size)
    lm_small_model = lm_embed.LanguageModelEmbed(embed_size=small_embed_size)

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir_path = os.path.join(this_dir_path, '..', '..', 'checkpoints', 'embedding')
    checkpoint_path = os.path.join(checkpoint_dir_path, 'lm_128', 'refined_small_model.pt')

    model_checkpoint_state_dict = torch.load(checkpoint_path)
    lm_large_model.load_state_dict(model_checkpoint_state_dict)

    roberta_model_state_dict = lm_large_model.model.roberta.state_dict()
    lm_small_model.model.roberta.load_state_dict(roberta_model_state_dict)

    small_model_checkpoint_path = os.path.join(checkpoint_dir_path, 'lm_32', 'small_model.pt')
    small_model_state_dict = lm_small_model.state_dict()
    torch.save(small_model_state_dict, small_model_checkpoint_path)

if __name__ == '__main__':
    main()