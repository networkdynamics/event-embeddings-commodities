import os

import pandas as pd
from tqdm import tqdm

from seldonite import sources, collect, run

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    repo_path = os.path.join(this_dir_path, '..', '..')
    commodities_data_path = os.path.join(repo_path, 'data', 'commodity_data')

    master_url = 'k8s://https://10.140.16.25:6443'
    aws_access_key = os.environ['AWS_ACCESS_KEY']
    aws_secret_key = os.environ['AWS_SECRET_KEY']

    spark_conf = {
        'spark.kubernetes.authenticate.driver.serviceAccountName': 'ben-dev',
        'spark.kubernetes.driver.pod.name': 'seldonite-driver',
        'spark.driver.host': 'seldonite-driver',
        'spark.driver.port': '7078'
    }

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

    for commodity in tqdm(commodities):
        commodity_articles_path = os.path.join(commodities_data_path, f"{commodity}_articles.csv")
        commodity_articles_df = pd.read_csv(commodity_articles_path)
        commodity_urls = commodity_articles_df['url'].values.tolist()

        cc_source = sources.news.CommonCrawl(aws_access_key, aws_secret_key)
        collector = collect.Collector(cc_source)
        collector.from_urls(commodity_urls) \
                 .get_features(['url', 'meta_keywords'])

        runner = run.Runner(collector, master_url=master_url, num_executors=11, executor_cores=4, executor_memory='48g', driver_memory='64g', spark_conf=spark_conf)
        commodity_keywords_df = runner.to_pandas()

        commodity_keywords_path = os.path.join(commodities_data_path, f"{commodity}_article_keywords.csv")
        commodity_keywords_df.to_csv(commodity_keywords_path)

if __name__ == '__main__':
    main()