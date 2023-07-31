import os

from scripts.commodity_prediction import forecasting

def main():
    commodity = 'crude_oil'
    suffix = 'lm_32_embed'
    days_ahead = 1
    seq_len = 50
    batch_size = 16

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    root_dir_path = os.path.join(this_dir_path, '..', '..')
    checkpoint_path = os.path.join(root_dir_path, 'checkpoints', 'commodity')
    model_checkpoints_dir = os.path.join(checkpoint_path, commodity, str(days_ahead), suffix)

    train_data, val_data, test_data, article_embeddings = forecasting.get_datasets(commodity, suffix, days_ahead, seq_len)
    train_dataloader, val_dataloader, test_dataloader = forecasting.get_dataloaders(train_data, val_data, test_data, batch_size)

    if not os.path.exists(os.path.dirname(model_checkpoints_dir)):
        os.makedirs(os.path.dirname(model_checkpoints_dir))

    creator = forecasting.ModelCreator(article_embeddings)
    model_params = {
        'hidden_size': 31, 'dropout': 0.14495244512786187, 'hidden_continuous_size': 19, 'attention_head_size': 3, 'learning_rate': 0.0013878906358348844
    }
    trainer_params = {
        'gradient_clip_val': 90
    }
    model = creator.get_model(train_data, **model_params)

    trainer = forecasting.get_trainer(model_checkpoints_dir, model=model, **trainer_params)

    forecasting.train(trainer, model, train_dataloader, val_dataloader)

if __name__ == '__main__':
    main()