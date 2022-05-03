import datetime
import json
import os

from seldonite import source
from seldonite import collect


def main():

    # sites to search
    sites = [ 'apnews.com', 'bbc.com', 'reuters.com', 'bloomberg.com' ]

    # master node
    master_url = 'k8s://https://10.140.16.25:6443'

    # fetch common crawl URLs
    cc_source = source.CommonCrawl(master_url=master_url)
    df = cc_source.query_index(f"""SELECT url_host_registered_domain,
                                      crawl,
                                      COUNT(*) AS count
                                   FROM ccindex 
                                   WHERE subset = 'warc' AND url_host_registered_domain IN ({",".join([f"'{site}'" for site in sites])})
                                   GROUP BY url_host_registered_domain, crawl""")

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir_path, '..', 'data')

    if not os.path.exists(data_path):
        os.mkdir(data_path)

    out_path = os.path.join(data_path, 'news_share', 'cc_news_share_data.csv')

    df.to_csv(out_path)

if __name__ == '__main__':
    main()