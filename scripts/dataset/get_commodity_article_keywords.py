import argparse
import os

import pandas as pd
from tqdm import tqdm

from seldonite import sources, collect, run, helpers

def main(args):
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    repo_path = os.path.join(this_dir_path, '..', '..')
    commodities_data_path = os.path.join(repo_path, 'data', 'commodity_data')

    master_url = 'k8s://https://10.140.16.25:6443'
    aws_access_key = args.aws_access_key
    aws_secret_key = args.aws_secret_key

    spark_conf = {
        'spark.kubernetes.authenticate.driver.serviceAccountName': 'ben-dev',
        'spark.kubernetes.driver.pod.name': 'seldonite-driver',
        'spark.driver.host': 'seldonite-driver',
        'spark.driver.port': '7078'
    }
    local_spark_conf = {
        'spark.driver.maxResultSize': '4g'
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

    blacklist = ['*/sports/*', '*rus.reuters*', '*fr.reuters*', '*br.reuters*', '*de.reuters*', '*es.reuters*', \
                 '*lta.reuters*', '*ara.reuters*', '*it.reuters*', '*ar.reuters*', '*blogs.reuters*', '*graphics.reuters*', \
                 '*jp.mobile.reuters*', '*live.special.reuters*', '*plus.reuters*', '*af.reuters*', \
                 '*in.reuters*', '*ru.reuters*', '*jp.reuters*', '*live.jp.reuters*', '*in.mobile.reuters*', \
                 '*br.mobile.reuters*', '*mx.reuters*', '*live.reuters*', '*cn.reuters*', '*agency.reuters*', \
                 '*widerimage.reuters*']

    crawls = helpers.utils.get_all_cc_crawls()
    not_crawls = ['CC-MAIN-2018-13', 'CC-MAIN-2018-09', 'CC-MAIN-2018-05', 
              'CC-MAIN-2017-51', 'CC-MAIN-2017-47', 'CC-MAIN-2017-43', 'CC-MAIN-2017-39', 
              'CC-MAIN-2017-34', 'CC-MAIN-2017-30', 'CC-MAIN-2017-26', 'CC-MAIN-2017-22', 
              'CC-MAIN-2017-17', 'CC-MAIN-2017-13', 'CC-MAIN-2017-09']
    crawls = [crawl for crawl in crawls if crawl not in not_crawls]

    for commodity in tqdm(commodities):
        commodity_articles_path = os.path.join(commodities_data_path, f"{commodity}_articles.csv")
        commodity_articles_df = pd.read_csv(commodity_articles_path)
        commodity_urls = commodity_articles_df['url'].values.tolist()

        cc_source = sources.news.CommonCrawl(aws_access_key, aws_secret_key)
        crawl = 'CC-MAIN-2018-34'
        cc_source.set_crawls(crawl)
        collector = collect.Collector(cc_source)
        collector.from_urls(commodity_urls) \
                 .on_sites(['reuters.com']) \
                 .exclude_in_url(blacklist) \
                 .distinct() \
                 .get_features(['url', 'meta_keywords'])

        #runner = run.Runner(collector, master_url=master_url, num_executors=11, executor_cores=4, executor_memory='48g', driver_memory='64g', spark_conf={**spark_conf, **local_spark_conf})
        runner = run.Runner(collector, driver_cores=24, driver_memory='64g', python_executable='/home/ndg/users/bsteel2/miniconda3/envs/seldonite/bin/python', spark_conf=local_spark_conf)
        commodity_keywords_df = runner.to_pandas()

        commodity_keywords_path = os.path.join(commodities_data_path, f"{commodity}_article_keywords.csv")
        commodity_keywords_df.to_csv(commodity_keywords_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--aws-access-key')
    parser.add_argument('--aws-secret-key')
    args = parser.parse_args()

    main(args)