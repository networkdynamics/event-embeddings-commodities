import datetime
import json
import os

from seldonite import source
from seldonite import collect


def main():

    # sites to search
    sites = [ 'apnews.com', 'bbc.com', 'nytimes.com' ]
    
    # master node
    master_url = 'k8s://https://10.140.16.25:6443'

    # fetch common crawl URLs
    cc_source = source.CommonCrawl(master_url=master_url, crawl='latest')
    collector = collect.Collector(cc_source)
    cc_articles = collector.fetch(sites=sites, max_articles=None, url_only=True, disable_news_heuristics=True)

    cc_urls = [article.source_url for article in cc_articles]

    url_data = {}
    url_data['num_ap_urls'] = sum(1 for url in cc_urls if 'apnews.com' in url)
    url_data['num_bbc_urls'] = sum(1 for url in cc_urls if 'bbc.com' in url)
    url_data['num_nyt_urls'] = sum(1 for url in cc_urls if 'nytimes.com' in url)

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir_path, '..', 'data')

    if not os.path.exists(data_path):
        os.mkdir(data_path)

    out_path = os.path.join(data_path, 'cc_news_share_data.json')

    with open(out_path, 'w') as f:
        json.dump(url_data, f)

if __name__ == '__main__':
    main()