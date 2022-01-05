import datetime
import json
import os

from seldonite import source, collect

def main():
    # master node
    master_url = 'k8s://https://10.140.16.25:6443'
    cc_source = source.CommonCrawl(master_url=master_url)
    
    collector = collect.Collector(cc_source)

    sites = [ 'reuters.com' ]
    start_date = datetime.date(2021, 7, 1)
    end_date = datetime.date(2021, 7, 30)
    collector.in_date_range(start_date, end_date) \
             .on_sites(sites) \
             .limit_num_articles(500) \
             .only_political_articles()

    cc_articles = collector.fetch()

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir_path, '..', 'data')

    if not os.path.exists(data_path):
        os.mkdir(data_path)
    
    out_path = os.path.join(data_path, 'political_news.json')

    with open(out_path, 'w') as f:
        json.dump([article.to_dict() for article in cc_articles], f, indent=2)

if __name__ == '__main__':
    main()