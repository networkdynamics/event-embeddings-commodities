import datetime
import logging
import os

from seldonite import collect, sources
from seldonite.helpers import utils

def main():
    #master_url = 'k8s://kubernetes.default.svc'
    master_url = 'k8s://https://10.140.16.25:6443'
    db_connection_string = os.environ['MONGO_CONNECTION_STRING']
    #db_connection_string = 'mongodb://ndl-user:ndllab123@10.244.0.152:27017,10.244.0.154:27017,10.244.0.153:27017/?ssl=false'

    spark_conf = {
        'spark.kubernetes.authenticate.driver.serviceAccountName': 'ben-dev',
        'spark.kubernetes.driver.pod.name': 'seldonite-driver',
        'spark.driver.host': 'seldonite-driver',
        'spark.driver.port': '7078'
    }
    
    db_name = 'political_events'
    in_table = 'debug_news'
    out_table = 'news'
    cc_source = sources.MongoDB(db_connection_string, db_name, in_table)
    #crawl = 'CC-MAIN-2020-34'
    #cc_source.set_crawls(crawl)
    

    start_dates = [
        datetime.date(2010, 1, 1),
        datetime.date(2015, 1, 1),
        datetime.date(2018, 1, 1),
        datetime.date(2019, 1, 1),
        datetime.date(2020, 1, 1)
    ]
    end_dates = [
        datetime.date(2015, 1, 1),
        datetime.date(2018, 1, 1),
        datetime.date(2019, 1, 1),
        datetime.date(2020, 1, 1),
        datetime.date(2021, 1, 1)
    ]

    min_start_date = datetime.date(2015, 1, 1)
    max_end_date = datetime.date(2018, 1, 1)

    start_date = min_start_date
    end_date = max_end_date

    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(this_file_dir, 'test_send_database.log')
    logging.basicConfig(filename=log_path, level=logging.INFO)

    # while end_date - start_date >= datetime.timedelta(days=1):
    #     try:
    #         collector = collect.Collector(cc_source, master_url=master_url, spark_conf=spark_conf)
    #         collector.on_sites(['reuters.com']) \
    #                 .exclude_in_url(['*/sports/*', '*rus.reuters*']) \
    #                 .in_language(lang='eng') \
    #                 .only_political_articles() \
    #                 .in_date_range(start_date, end_date)

    #         collector.fetch().to_csv('./debug_political_news.csv')
    #         logging.info(str(start_date) + ' ' + str(end_date) + ' is good')
    #         start_date = end_date
    #         end_date = start_date + interval
    #     except:
    #         logging.info(str(start_date) + ' ' + str(end_date) + ' is bad')
    #         interval = ((end_date - start_date) / 2)
    #         end_date = start_date + interval
    #collector.send_to_database(db_connection_string, db_name, out_table)

    start_date = datetime.date(2018, 1, 1)
    end_date = datetime.date(2020, 1, 1)

    collector = collect.Collector(cc_source, master_url=master_url, spark_conf=spark_conf)
    collector.on_sites(['reuters.com']) \
            .exclude_in_url(['*/sports/*', '*rus.reuters*']) \
            .in_language(lang='eng') \
            .only_political_articles()
            #.in_date_range(start_date, end_date)

    collector.send_to_database(db_connection_string, db_name, out_table)
    #collector.fetch().to_csv('./debug_political_news.csv')

if __name__ == '__main__':
    main()