import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir_path, '..', '..', 'data')
    relations_path = os.path.join(data_path, 'relations')

    path_keyword = 'lm_small_embed'#'0521_news2vec_embeds'
    very_high_titles = [
        'Russia deploys more surface-to-air missiles in Crimean build-up', 
        "Rebels push west as air strikes hit Gaddafi forces",
        "U.S. House may vote within days on tighter North Korea sanctions",
        "NATO says Libya airstrikes cripple Gaddafi's forces",
        "Two rockets fall inside Iraqi air base housing U.S. troops: security sources",
        "Shots fired as Russian troops force their way into Ukrainian base in Crimea",
        "Russian forces seize two Ukrainian bases in Crimea"
    ]
    high_titles = [
        
        "Canadian PM says mosque shooting a 'terrorist attack on Muslims'", 
        "Clinton: Iran moving toward military dictatorship",
        "U.S. accuses Russian spies of 2016 election hacking as summit looms",
        "WRAPUP 6-Putin completes Crimea's annexation, Russia investors take fright",
        "College student from California studying abroad killed in Paris attacks",
        "France's Macron seeks extended emergency powers after Manchester attack",
        "Islamic State leader killed in air strike linked to two Paris attackers",
        "Suicide bombers attack 2 police stations in Iraq's Samarra - sources"
        ]
    mid_high_titles = [
        "U.S. eyeing plan for fifth brigade in Afghanistan",
        "Iraq's Kurds have right to sell oil while squeezed by Baghdad: MP",
        "As Tibetan self-immolations rise, Beijing tightens grip",
        "Military planners prepare for war in Mali",
        "U.S. judge orders former Trump aides to stay under home arrest",
        "Analysis: After U.S. embassy attack, West uneasy over Saleh's role",
        "Egypt president promises to fight chaos before pro-Mursi rallies",
        "Egyptian crackdown risks spreading instability abroad, Islamist says",
        "U.N. nuclear inspectors in Iran, no sign of Parchin visit"
    ]
    low_titles = [
        'Britain says in talks with Iran about reopening embassies', 
        "Africa's emerging middle class drives growth and democracy", 
        "Bush to visit biblical site on peacemaking tour", 
        '''"At U.N., Congo's Kabila vows 'peaceful, credible' elections"''', 
        '''"U.S., Japan to cooperate on energy, infrastructure investment: Treasury"''',
        "U.S.-led troops withdraw from Iraq's Taji base",
        "South, North Korea reopen hotlines as leaders seek to rebuild ties",
        "FACTBOX - North, South Korea pledge peace, prosperity",
        "Obama pledges diplomacy with Iran; no Rouhani meeting"
        ]
    mid_low_titles = [
        "Syria envoy in Damascus, but prospects for peace talks dim",
        "Analysis: Syria peace talks look doomed in advance",
        "'New turning point' for ties, China tells Philippines visitors",
        "Cuba's Raul Castro meets with U.S. congressional delegation",
        "Chinese envoy to U.S. urges stable commercial ties despite trade conflicts"
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
        "Why bond investors are willing to bet on money-losing Pemex after oil price crash",
        "Factbox - Canada's Trudeau shuffles cabinet, names new foreign minister",
        "Earthquake rattles Afghanistan, Pakistan and Indian Kashmir",
        "UPDATE 2-Cracker Barrel Q1 beats Street, ups FY EPS view",
        "COLUMN-Blood on the floor as market hunts WTI longs: Kemp",
        "UPDATE 2-EIA raises 2010 OPEC, non-OPEC oil output forecast",
        "UPDATE 1-Enerplus reports Q1 profit on higher crude oil prices",
        "Whiting Petroleum Corporation Announces First Quarter 2013 Financial and Operating Results",
        "Olympics-Diving-Men's 10m platform preliminary round results",
        '"Yuan weakens, safe havens gain on Chinese virus concerns"',
        "Peru's dynamic first lady has presidential aura",
        "China plans tougher quality standards for coal to tackle pollution",
        "Eight in race for top job at World Trade Organization",
        "IBM unveils new server model to tackle big data, analytics",
        "U.S. House passes $8.3 billion bill to battle coronavirus; Senate vote due Thursday"
    ]

    us_embed_path = os.path.join(relations_path, f'united_states_{path_keyword}.csv')
    df = pd.read_csv(us_embed_path)
    df = df.rename(columns={'embed': 'embedding'})
    df = df[['publish_date', 'title', 'embedding']]
    df['publish_date'] = pd.to_datetime(df['publish_date'])
    df['embedding'] = df['embedding'].str.strip('[]').apply(lambda x: np.fromstring(x, sep=' '))

    very_high_embeds_df = df[df['title'].isin(very_high_titles)]
    high_embeds_df = df[df['title'].isin(high_titles)]
    mid_high_embeds_df = df[df['title'].isin(mid_high_titles)]
    low_embeds_df = df[df['title'].isin(low_titles)]
    mid_low_embeds_df = df[df['title'].isin(mid_low_titles)]
    neutral_embeds_df = df[df['title'].isin(neutral_titles)]

    very_high_embeds = np.stack(very_high_embeds_df['embedding'].values)
    high_embeds = np.stack(high_embeds_df['embedding'].values)
    mid_high_embeds = np.stack(mid_high_embeds_df['embedding'].values)
    low_embeds = np.stack(low_embeds_df['embedding'].values)
    mid_low_embeds = np.stack(mid_low_embeds_df['embedding'].values)
    neutral_embeds = np.stack(neutral_embeds_df['embedding'].values)
    
    
    
    model_keyword = 'logistic'

    if model_keyword == 'regression':
        X = np.concatenate([very_high_embeds, high_embeds, mid_high_embeds, low_embeds, mid_low_embeds, neutral_embeds])
        #X = np.concatenate([high_embeds, mid_high_embeds, low_embeds, mid_low_embeds, neutral_embeds])
        y = np.array([10] * len(very_high_embeds) + [5] * len(high_embeds) + [1] * len(mid_high_embeds) + [-1] * len(low_embeds) + [-0.5] * len(mid_low_embeds) + [0] * len(neutral_embeds)).reshape(-1,1)
        #y = np.array([5] * len(high_embeds) + [1] * len(mid_high_embeds) + [-1] * len(low_embeds) + [-0.5] * len(mid_low_embeds) + [0] * len(neutral_embeds)).reshape(-1,1)
        reg = LinearRegression().fit(X,y)

        def get_prediction(embed):
            return reg.predict(embed.reshape(1,-1))[0][0]

    elif model_keyword == 'logistic':
        X = np.concatenate([very_high_embeds, high_embeds, mid_high_embeds, low_embeds, mid_low_embeds, neutral_embeds])
        #X = np.concatenate([high_embeds, mid_high_embeds, low_embeds, mid_low_embeds, neutral_embeds])
        y = np.array([1] * len(very_high_embeds) + [1] * len(high_embeds) + [1] * len(mid_high_embeds) + [-1] * len(low_embeds) + [-1] * len(mid_low_embeds) + [0] * len(neutral_embeds)).reshape(-1,1)
        #y = np.array([5] * len(high_embeds) + [1] * len(mid_high_embeds) + [-1] * len(low_embeds) + [-0.5] * len(mid_low_embeds) + [0] * len(neutral_embeds)).reshape(-1,1)
        reg = LogisticRegression().fit(X,y)

        def get_prediction(embed):
            return reg.predict(embed.reshape(1,-1))[0]


    #article_embed_paths = [os.path.join(relations_path, filename) for filename in os.listdir(relations_path) if path_keyword in filename]
    article_embed_paths = [us_embed_path]

    for article_embed_path in article_embed_paths:
        print(f"Getting index for {os.path.basename(article_embed_path)}")
        df = pd.read_csv(article_embed_path)
        df = df.rename(columns={'embed': 'embedding'})
        df = df[['publish_date', 'title', 'embedding']]
        df['publish_date'] = pd.to_datetime(df['publish_date'])
        df['embedding'] = df['embedding'].str.strip('[]').apply(lambda x: np.fromstring(x, sep=' '))
        df['index'] = df['embedding'].apply(get_prediction)
        df = df[['publish_date', 'title', 'index']]
        df.to_csv(article_embed_path.replace(path_keyword, f'{path_keyword}_threat_{model_keyword}'), index=False)


if __name__ == '__main__':
    main()