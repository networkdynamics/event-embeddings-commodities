import datetime
import json
import os

from seldonite import source, collect

def main():
    # master node
    master_url = 'k8s://https://10.140.16.25:6443'
    cc_source = source.NewsCrawl(master_url=master_url)
    
    collector = collect.Collector(cc_source)

    start_date = datetime.date(2021, 7, 1)
    end_date = datetime.date(2021, 7, 30)
    collector.in_date_range(start_date, end_date) \
             .limit_num_articles(50000) \
             .url_only()

    cc_articles = collector.fetch(disable_news_heuristics=True)

    cc_urls = [article.source_url for article in cc_articles]

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir_path, '..', 'data')

    if not os.path.exists(data_path):
        os.mkdir(data_path)
    
    out_path = os.path.join(data_path, 'newscrawl_urls.json')

    with open(out_path, 'w') as f:
        json.dump(cc_urls, f)

if __name__ == '__main__':
    main()