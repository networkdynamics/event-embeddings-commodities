import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_forecasting.utils import to_list

import forecasting

def string_to_dict(string, pattern):
    regex = re.sub(r'{(.+?)}', r'(?P<_\1>.+)', pattern)
    search = re.search(regex, string)
    if not search:
        return {}
    values = list(search.groups())
    keys = re.findall(r'{(.+?)}', pattern)
    _dict = dict(zip(keys, values))
    return _dict

def plot_prediction(
        model,
        x,
        out,
        target_normalizer,
        show_future_observed = True,
        ax=None,
        quantiles_kwargs = {},
        prediction_kwargs = {},
    ):
    # all true values for y of the first sample in batch
    encoder_targets = to_list(x["encoder_target"])
    decoder_targets = to_list(x["decoder_target"])

    y_raws = to_list(out["prediction"])  # raw predictions - used for calculating loss
    y_hats = to_list(model.to_prediction(out, **prediction_kwargs))
    y_quantiles = to_list(model.to_quantiles(out, **quantiles_kwargs))

    # for each target, plot
    figs = []
    for y_raw, y_hat, y_quantile, encoder_target, decoder_target in zip(
        y_raws, y_hats, y_quantiles, encoder_targets, decoder_targets
    ):
        y = torch.cat([encoder_target[0], decoder_target[:,0]])
        # move predictions to cpu
        y_hat = y_hat.detach().cpu()[:, -1]
        y_quantile = y_quantile.detach().cpu()[:, -1]
        y_raw = y_raw.detach().cpu()[:, -1]

        # move to cpu
        y = y.detach().cpu()

        # inverse transform
        y = target_normalizer.inverse_transform(y)
        y_hat = target_normalizer.inverse_transform(y_hat)
        y_quantile = target_normalizer.inverse_transform(y_quantile)
        y_raw = target_normalizer.inverse_transform(y_raw)

        # create figure
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        n_pred = y_hat.shape[0]
        x_obs = np.arange(-(y.shape[0] - n_pred), 0)
        x_pred = np.arange(n_pred)
        prop_cycle = iter(plt.rcParams["axes.prop_cycle"])
        obs_color = next(prop_cycle)["color"]
        pred_color = next(prop_cycle)["color"]
        # plot observed history
        if len(x_obs) > 0:
            if len(x_obs) > 1:
                plotter = ax.plot
            else:
                plotter = ax.scatter
            plotter(x_obs, y[:-n_pred], label="observed", c=obs_color)
        if len(x_pred) > 1:
            plotter = ax.plot
        else:
            plotter = ax.scatter

        # plot observed prediction
        if show_future_observed:
            plotter(x_pred, y[-n_pred:], label=None, c=obs_color)

        # plot prediction
        plotter(x_pred, y_hat, label="predicted", c=pred_color)

        # plot predicted quantiles
        plotter(x_pred, y_quantile[:, y_quantile.shape[1] // 2], c=pred_color, alpha=0.15)
        for i in range(y_quantile.shape[1] // 2):
            if len(x_pred) > 1:
                ax.fill_between(x_pred, y_quantile[:, i], y_quantile[:, -i - 1], alpha=0.15, fc=pred_color)
            else:
                quantiles = torch.tensor([[y_quantile[0, i]], [y_quantile[0, -i - 1]]])
                ax.errorbar(
                    x_pred,
                    y[[-n_pred]],
                    yerr=quantiles - y[-n_pred],
                    c=pred_color,
                    capsize=1.0,
                )

        ax.set_xlabel("Time index")
        fig.legend()
        figs.append(fig)

    # return multiple of target is a list, otherwise return single figure
    if isinstance(x["encoder_target"], (tuple, list)):
        return figs
    else:
        return fig

def main():

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    root_dir_path = os.path.join(this_dir_path, '..', '..')
    fig_dir_path = os.path.join(root_dir_path, 'figs')
    checkpoint_path = os.path.join(root_dir_path, 'checkpoints', 'commodity')

    batch_size = 16
    seq_len = 50
    all = False
    method = 'lm_32_embed'
    variable = 'commodities'
    plot_type = 'by_value'

    if method == 'sentiment':
        suffix = 'sentiment'
    elif method == 'news2vec':
        suffix = '0521_news2vec_embeds'
    elif method == 'lm_embed':
        suffix = 'lm_both_embed'
    elif method =='lm_32_embed':
        suffix = 'lm_32_embed'
    elif method =='lm_context_embed':
        suffix = 'lm_context_embed'
    elif method =='lm_graph_embed':
        suffix = 'lm_graph_embed'
    elif method == 'constant':
        suffix = 'constant'
    elif method == 'no_feature':
        suffix = 'no_feature'

    scores = {}

    if variable == 'commodities':

        days_ahead = 1
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
        
        for commodity in commodities:
            print(f"Testing {commodity}") 

            model_checkpoints_dir = os.path.join(checkpoint_path, 'all' if all else commodity, str(days_ahead), method)

            df, article_embeddings, get_split, article_index_cols = forecasting.get_dataframes(commodity, suffix, splits=[0.7, 0.15, 0.15])
            train_data, val_data, test_data, article_embeddings = forecasting._get_datasets(commodity, suffix, days_ahead, seq_len, df, article_embeddings, get_split, article_index_cols)
            train_dataloader, val_dataloader, test_dataloader = forecasting.get_dataloaders(train_data, val_data, test_data, batch_size)

            if suffix == 'constant':
                model = forecasting.get_constant_model(train_data)
            else:
                model_creator = forecasting.ModelCreator(article_embeddings)
                model = model_creator.get_model(train_data)

            best_model_loss = 100000
            best_model_path = None

            for file_name in os.listdir(model_checkpoints_dir):
                match = re.search('\d{1,2}.\d{2,4}', file_name)
                if match:
                    val_loss = float(match.group(0))
                    if val_loss < best_model_loss:
                        best_model_loss = val_loss
                        best_model_path = os.path.join(model_checkpoints_dir, file_name)
            
            best_model = model.load_from_checkpoint(best_model_path)

            # fix embedding sizes
            best_model.hparams.embedding_sizes = model.hparams.embedding_sizes
            best_model.hparams.embedding_labels = model.hparams.embedding_labels

            if plot_type == 'raw':
                # raw predictions are a dictionary from which all kind of information including quantiles can be extracted
                raw_predictions = best_model.predict(test_dataloader, mode="raw", return_x=True)
                fig = plot_prediction(best_model, raw_predictions[1], raw_predictions[0], test_data.target_normalizer)

                fig_path = os.path.join(fig_dir_path, f"{commodity}_{method}.png")
                if not os.path.exists(os.path.dirname(fig_path)):
                    os.mkdir(os.path.dirname(fig_path))
                fig.savefig(fig_path)

            elif plot_type == 'by_value':
                predictions = best_model.predict(val_dataloader, return_x=True)
                predictions_vs_actuals = best_model.calculate_prediction_actual_by_variable(predictions[1], predictions[0])
                figs = best_model.plot_prediction_actual_by_variable(predictions_vs_actuals)

                var_names = ['time_index', 'price_diff', 'volume', 'price_type', 'month']
                for var_name in var_names:
                    fig_path = os.path.join(fig_dir_path, f"{commodity}_{method}_{var_name}.png")
                    if not os.path.exists(os.path.dirname(fig_path)):
                        os.mkdir(os.path.dirname(fig_path))
                    figs[var_name].savefig(fig_path)

if __name__ == '__main__':
    main()