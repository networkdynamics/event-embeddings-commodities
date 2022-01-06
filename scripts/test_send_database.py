import argparse
import os

from seldonite import collect, source
from seldonite.helpers import utils

def main():
    #master_url = 'k8s://kubernetes.default.svc'
    master_url = 'k8s://https://10.140.16.25:6443'
    #db_connection_string = os.environ['MONGO_CONNECTION_STRING']
    db_connection_string = 'mongodb://ndl-user:ndllab123@10.244.0.152:27017,10.244.0.154:27017,10.244.0.153:27017/?ssl=false'
    
    db_name = 'political_events'
    db_table = 'news'
    crawls = utils.get_all_cc_crawls()
    cc_source = source.CommonCrawl()
    cc_source.set_crawls(crawls[0])
    collector = collect.Collector(cc_source, master_url=master_url)
    collector.limit_num_articles(100) \
             .on_sites(['reuters.com']) \
             .exclude_in_path(['/sports/']) \
             .in_language(lang='eng') \
             .url_only()

    #collector.fetch().to_csv('./political_news.csv')
    collector.send_to_database(db_connection_string, db_name, db_table)

if __name__ == '__main__':
    main()