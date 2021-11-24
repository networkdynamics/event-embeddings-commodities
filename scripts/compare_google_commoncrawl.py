import argparse
import json
import os

from seldonite import source
from seldonite import collect


def main(args):

    # sites to search
    sites = [ 'apnews.com' ]
        
    # fetch common crawl URLs
    cc_source = source.CommonCrawl(master_url=args.master_url, sites=sites)
    
    collector = collect.Collector(cc_source)
    cc_articles = collector.fetch(max_articles=100, url_only=True)

    cc_urls = set(article.source_url for article in cc_articles)

    # fetch google urls
    google_source = source.Google(dev_key=args.dev_key, engine_id=args.engine_id, sites=sites)

    collector = collect.Collector(google_source)
    google_articles = collector.fetch(max_articles=100, url_only=True)

    google_urls = set(article.source_url for article in google_articles)

    # find common urls
    common_urls = google_urls.intersection(cc_urls)

    url_data = {
        'cc_urls': cc_urls,
        'google_urls': google_urls,
        'common_urls': common_urls,
        'num_common': len(common_urls)
    }

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir_path, '..', 'data')
    out_path = os.path.join(data_path, 'cc_coverage_data.json')

    with open(out_path, 'w') as f:
        json.dump(url_data, f)

if __name__ == '__main__':
    main()