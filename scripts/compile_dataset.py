import argparse

from seldonite import collect, source
from seldonite.helpers import utils

def main(args):
    master_url = 'k8s://https://10.140.16.25:6443'
    db_connection_string = args.connectionstring
    db_name = 'political_events'
    db_table = 'news'
    crawls = utils.get_all_cc_crawls()
    for crawl in crawls:
        print(f"Starting pull for crawl: {crawl}")
        cc_source = source.CommonCrawl()
        cc_source.set_crawls(crawl)
        collector = collect.Collector(cc_source, master_url=master_url)
        collector.only_political_articles() \
                 .on_sites(['reuters.com']) \
                 .exclude_in_path(['/sports/']) \
                 .in_language(lang='eng')

        collector.send_to_database(db_connection_string, db_name, db_table)
        print(f"Finished pull for crawl: {crawl}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--connectionstring')
    args = parser.parse_args()
    main(args)