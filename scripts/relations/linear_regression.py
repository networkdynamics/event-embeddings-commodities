import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir_path, '..', '..', 'data')
    relations_path = os.path.join(data_path, 'relations')

    path_keyword = 'lm_embed'
    high_titles = [
        'Russia deploys more surface-to-air missiles in Crimean build-up', 
        "Canadian PM says mosque shooting a 'terrorist attack on Muslims'", 
        "Clinton: Iran moving toward military dictatorship",
        "Two rockets fall inside Iraqi air base housing U.S. troops: security sources",
        "U.S. accuses Russian spies of 2016 election hacking as summit looms",
        "WRAPUP 6-Putin completes Crimea's annexation, Russia investors take fright",
        "Rebels push west as air strikes hit Gaddafi forces",
        "College student from California studying abroad killed in Paris attacks",
        "U.S. House may vote within days on tighter North Korea sanctions"
        ]
    mid_high_titles = [
        "U.S. eyeing plan for fifth brigade in Afghanistan"
    ]
    low_titles = [
        'Britain says in talks with Iran about reopening embassies', 
        "Africa's emerging middle class drives growth and democracy", 
        "Bush to visit biblical site on peacemaking tour", 
        '''"At U.N., Congo's Kabila vows 'peaceful, credible' elections"''', 
        '''"U.S., Japan to cooperate on energy, infrastructure investment: Treasury"''',
        "U.S.-led troops withdraw from Iraq's Taji base"
        ]
    neutral_titles = [
        "U.S. STOCK INDEX FUTURES EXTEND FALL; DOW, S&P FUTURES DOWN 0.65 PCT, NASDAQ FUTURES DOWN 0.85 PCT",
        "TABLE-Malaysia economic indicators - Aug 9",
        "FACTBOX: Electoral votes by state in Tuesday's election",
        "Fitch Corrects Certain Issue Level Ratings for KfW",
        "BRIEF-Singapore Press Holdings announces divestment of stake in 701Search Pte Ltd",
        "BRIEF-Lithium Americas' subsidiary Hectatone announces royalty agreement with Delmon",
        "ARCELORMITTAL REPORTS SECOND QUARTER 2013 AND HALF YEAR 2013 RESULTS",
        "BRIEF-Cowen and CEFC China announce mutual agreement to withdraw from filing with the committee on foreign investment in the United States (CFIUS)",
        "Magnitude 6.2 quake hits east of Vanuatu: USGS",
        "ADVISORY: UBS Singapore traders story withdrawn",
        "Why bond investors are willing to bet on money-losing Pemex after oil price crash"
    ]

    us_embed_path = os.path.join(relations_path, 'united_states_lm_embed.csv')
    df = pd.read_csv(us_embed_path)
    df = df.rename(columns={'embed': 'embedding'})
    df = df[['publish_date', 'title', 'embedding']]
    df['publish_date'] = pd.to_datetime(df['publish_date'])
    df['embedding'] = df['embedding'].str.strip('[]').apply(lambda x: np.fromstring(x, sep=' '))

    high_embeds_df = df[df['title'].isin(high_titles)]
    mid_high_embeds_df = df[df['title'].isin(mid_high_titles)]
    low_embeds_df = df[df['title'].isin(low_titles)]
    neutral_embeds_df = df[df['title'].isin(neutral_titles)]
    high_embeds = np.stack(high_embeds_df['embedding'].values)
    mid_high_embeds = np.stack(mid_high_embeds_df['embedding'].values)
    low_embeds = np.stack(low_embeds_df['embedding'].values)
    neutral_embeds = np.stack(neutral_embeds_df['embedding'].values)
    
    X = np.concatenate([high_embeds, mid_high_embeds, low_embeds, neutral_embeds])
    y = np.array([1] * len(high_embeds) + [0.5] * len(mid_high_embeds) + [-1] * len(low_embeds) + [0] * len(neutral_embeds)).reshape(-1,1)
    reg = LinearRegression().fit(X,y)

    article_embed_paths = [os.path.join(relations_path, filename) for filename in os.listdir(relations_path) if path_keyword in filename]

    for article_embed_path in article_embed_paths:
        print(f"Getting index for {os.path.basename(article_embed_path)}")
        df = pd.read_csv(article_embed_path)
        df = df.rename(columns={'embed': 'embedding'})
        df = df[['publish_date', 'title', 'embedding']]
        df['publish_date'] = pd.to_datetime(df['publish_date'])
        df['embedding'] = df['embedding'].str.strip('[]').apply(lambda x: np.fromstring(x, sep=' '))
        df['index'] = df['embedding'].apply(lambda embed: reg.predict(embed.reshape(1,-1))[0][0])
        df = df[['publish_date', 'title', 'index']]
        df.to_csv(article_embed_path.replace(path_keyword, 'threat_regression'), index=False)


if __name__ == '__main__':
    main()