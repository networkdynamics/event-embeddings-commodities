import json
import os
import re

import pandas as pd

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', '..', 'data', 'results', 'commodity')

    model = 'LSTM'
    suffix = 'lm_32_embed'

    all_results = {}

    for file_name in os.listdir(data_dir_path):
        if not file_name.endswith(f'_{suffix}_{model}_granger.json'):
            continue

        with open(os.path.join(data_dir_path, file_name), 'r') as f:
            results = json.load(f)

        commodity = re.search(f'[a-z_]*(?=_{suffix}_{model})', file_name).group(0)

        formatted_results = {}
        for price_type in results:
            for lag_order in results[price_type]:
                if lag_order not in formatted_results:
                    formatted_results[lag_order] = {}
                formatted_results[lag_order][price_type] = results[price_type][lag_order]

        all_results[commodity] = formatted_results

    final_results = {}
    for commodity in all_results:
        final_results[commodity] = {}
        for lag_order in all_results[commodity]:
            min_p_value = min(result['p_value'] for result in all_results[commodity][lag_order].values())
            max_p_value = max(result['p_value'] for result in all_results[commodity][lag_order].values())
            final_results[commodity][lag_order] = f"{min_p_value} - {max_p_value}"

    df = pd.DataFrame(final_results).reset_index().rename(columns={'index': 'lag_order'})

    commodity_order = [
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
    df = df[['lag_order'] + commodity_order]

    df.to_csv(os.path.join(data_dir_path, f'{suffix}_{model}_granger.csv'), index=False)

if __name__ == '__main__':
    main()