import argparse
import datetime
import json
import os

from seldonite import source
from seldonite import collect


def main(args):

    # sites to search
    sites = [ 'bbc.com', 'bbc.co.uk' ]

    # master node
    master_url = 'k8s://https://10.140.16.25:6443'

    # fetch google urls
    google_source = source.Google(dev_key=args.dev_key, engine_id=args.engine_id)

    collector = collect.Collector(google_source)

    start_date = datetime.date(2021, 8, 1)
    end_date = datetime.date(2021, 8, 31)
    collector.in_date_range(start_date, end_date)

    #keywords = ['election']
    keywords = ['election', 'president', 'war', 'government', 'border']
    keyword_urls = {}
    for keyword in keywords:
        cache_path = os.path.join('.', 'data', 'news_coverage', f'google_news_urls_bbc_{keyword}_aug_2021.json')
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                keyword_urls[keyword] = set(json.load(f))
            continue

        collector.by_keywords([keyword]) \
                 .on_sites(sites) \
                 .limit_num_articles(100) \
                 .url_only()

        google_articles = collector.fetch(disable_news_heuristics=True)

        keyword_urls[keyword] = set(article.source_url for article in google_articles)

        if not os.path.exists(cache_path):
            with open(cache_path, 'w') as f:
                json.dump(list(keyword_urls[keyword]), f)

    # fetch common crawl URLs
    cc_source = source.NewsCrawl(master_url=master_url)
    
    collector = collect.Collector(cc_source)

    start_date = datetime.date(2021, 8, 1)
    end_date = datetime.date(2021, 8, 31)
    collector.in_date_range(start_date, end_date) \
             .on_sites(sites) \
             .url_only()

    cc_articles = collector.fetch(disable_news_heuristics=True)

    cc_urls = set(article.source_url for article in cc_articles)

    # find common urls
    url_data = {}
    avg_num_in_common = 0
    for keyword, urls in keyword_urls.items():
        # TODO divide by number of urls from google
        num_in_common = len(urls.intersection(cc_urls))
        avg_num_in_common += num_in_common
        url_data[keyword] = num_in_common

    url_data['avg_num_common'] = avg_num_in_common / len(keyword_urls)
    url_data['num_cc_urls'] = len(cc_urls)

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir_path, '..', 'data', 'news_coverage')

    if not os.path.exists(data_path):
        os.mkdir(data_path)
    
    out_path = os.path.join(data_path, 'newscrawl_coverage_data_bbc.json')

    with open(out_path, 'w') as f:
        json.dump(url_data, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev-key')
    parser.add_argument('--engine-id')
    args = parser.parse_args()

    main(args)