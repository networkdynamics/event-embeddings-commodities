import collections
import os
from urllib.parse import urlparse

from adjustText import adjust_text
import matplotlib.pyplot as plt
import newspaper
import pandas as pd
import tqdm

def is_interesting_section(section):
    boring_sections = ['article']
    if section in boring_sections:
        return False

    if section.isnumeric():
        return False

    return True

def get_categories_from_url(url):
    full_path = urlparse(url).path
    site_path, page_name = os.path.split(full_path)

    # get sections
    rest_site_path, last_section = os.path.split(site_path)
    sections = [last_section]
    while rest_site_path != '/':
        rest_site_path, section = os.path.split(rest_site_path)
        sections = [section] + sections

    # filter out common sections
    filtered_sections = [section for section in sections if is_interesting_section(section)]

    return filtered_sections

def get_categories(commodity, commodities_data_path):
    commodity_articles_path = os.path.join(commodities_data_path, f"{commodity}_articles.csv")
    commodity_articles_df = pd.read_csv(commodity_articles_path)
    commodity_categories = [category for url in tqdm.tqdm(commodity_articles_df['url']) for category in get_categories_from_url(url)]
    return commodity_categories, len(commodity_articles_df)

def str_to_list(stri):
    return [item.strip()[1:-1] for item in stri[1:-1].split(',')]

def get_keywords(commodity, commodities_data_path):
    commodity_articles_path = os.path.join(commodities_data_path, f"{commodity}_article_keywords.csv")
    commodity_articles_df = pd.read_csv(commodity_articles_path)
    commodity_articles_df['meta_keywords'] = commodity_articles_df['meta_keywords'].apply(str_to_list)
    commodity_articles_df['meta_keywords'] = commodity_articles_df['meta_keywords'].apply(lambda ks: [k.lower() for k in ks])
    commodity_categories = [keyword for keywords in commodity_articles_df['meta_keywords'].values for keyword in keywords if keyword != '']
    return commodity_categories, len(commodity_articles_df)

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    repo_path = os.path.join(this_dir_path, '..', '..')
    commodities_data_path = os.path.join(repo_path, 'data', 'commodity_data')
    accuracies_path = os.path.join(commodities_data_path, 'accuracies.csv')

    df = pd.read_csv(accuracies_path)
    df = df.set_index('method').transpose().rename_axis('commodity')

    df['embedding'] = df[['news2vec', 'newsformer128', 'newsformer32']].min(axis=1)
    df['diff'] = (df['sentiment'] - df['embedding']) / df['embedding']

    variable = 'article_category'

    if variable == 'num_articles':
        independent_variable = 'num_articles'
        fig_name = f'{variable}_compare.png'
        plt.scatter(df[independent_variable], df['diff'])
        plt.xscale('log')
    elif variable == 'trading_volume':
        independent_variable = 'volume'
        fig_name = f'{variable}_compare.png'
        df['volume'] = None
        for commodity in df.index:
            commodity_data_path = os.path.join(commodities_data_path, f"{commodity}.csv")
            commodity_df = pd.read_csv(commodity_data_path)
            df.loc[commodity, 'volume'] = commodity_df['Volume'].mean()

        plt.scatter(df[independent_variable], df['diff'])
        plt.xscale('log')
    elif variable == 'article_category':
        independent_variable = 'categories'
        category_type = 'keywords'

        if category_type == 'path':
            cat_func = get_categories
        elif category_type == 'keywords':
            cat_func = get_keywords

        fig_name = f'{variable}_by_{category_type}_compare.png'

        df['categories'] = None
        category_scores = {}
        for commodity in df.index:
            commodity_categories, num_articles = cat_func(commodity, commodities_data_path)
            category_counter = collections.Counter(commodity_categories)
            commodity_top_categories = category_counter.most_common(10)
            acc_diff = df.loc[commodity, 'diff']
            for category, count in commodity_top_categories:
                if category in category_scores:
                    category_scores[category] = (category_scores[category][0] + (count / num_articles), category_scores[category][1] + acc_diff)
                else:
                    category_scores[category] = ((count / num_articles), acc_diff)

        remove_keywords = (
            'article', 'crude oil', 'wheat', 'corn', 'platinum', 'copper', 
            'gold', 'gasoline', 'palladium', 'silver', 'soybean', 'sugar',
            'platinum group metals', 'grains', 'oil and gas (trbc)', 'precious metals', 
            'metals markets', 'agricultural markets', 'all precious metals and minerals',
            'precious metals and minerals (trbc)', 'metals and mining (trbc)', 'mining',
            'refined products', 'energy (legacy)', 'us'
        )
        for keyword in remove_keywords:
            category_scores.pop(keyword, None)

        texts = [plt.text(scores[0], scores[1], category, ha='center', va='center') for category, scores in category_scores.items()]
        
        min_x_score = min(scores[0] for scores in category_scores.values())
        max_x_score = max(scores[0] for scores in category_scores.values())
        diff_x_score = max_x_score - min_x_score
        plt.xlim(min_x_score - (0.2*diff_x_score), max_x_score + (0.2*diff_x_score))
        min_y_score = min(scores[1] for scores in category_scores.values())
        max_y_score = max(scores[1] for scores in category_scores.values())
        diff_y_score = max_y_score - min_y_score
        plt.ylim(min_y_score - (0.2*diff_y_score), max_y_score + (0.2*diff_y_score))

        adjust_text(texts, expand_text=(1.05, 1.05))
    
    plt.axhline(y=0, linestyle='--')
    plt.xlabel(independent_variable.title().replace('_', ' '))
    plt.ylabel('Relative Accuracy Difference')

    fig_path = os.path.join(repo_path, 'figs', fig_name)
    plt.savefig(fig_path)
    
if __name__ == '__main__':
    main()