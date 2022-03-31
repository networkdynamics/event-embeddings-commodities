import logging
import os

from seldonite import collect, sources, run

def log(log_msg):
    print(log_msg)
    logging.info(log_msg)

def main():
    master_url = 'k8s://https://10.140.16.25:6443'
    db_connection_string = os.environ['MONGO_CONNECTION_STRING']

    spark_conf = {
        'spark.kubernetes.authenticate.driver.serviceAccountName': 'ben-dev',
        'spark.kubernetes.driver.pod.name': 'seldonite-driver',
        'spark.driver.host': 'seldonite-driver',
        'spark.driver.port': '7078'
    }

    db_name = 'political_events'
    db_table = 'reuters_news'

    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(this_file_dir, 'compile_dataset.log')
    logging.basicConfig(filename=log_path, level=logging.INFO)

    #crawls = utils.get_all_cc_crawls()
    crawls = ['CC-MAIN-2018-34', 'CC-MAIN-2018-30', 'CC-MAIN-2018-26', 'CC-MAIN-2018-22', 
              'CC-MAIN-2018-17', 'CC-MAIN-2018-13', 'CC-MAIN-2018-09', 'CC-MAIN-2018-05', 
              'CC-MAIN-2017-51', 'CC-MAIN-2017-47', 'CC-MAIN-2017-43', 'CC-MAIN-2017-39', 
              'CC-MAIN-2017-34', 'CC-MAIN-2017-30', 'CC-MAIN-2017-26', 'CC-MAIN-2017-22', 
              'CC-MAIN-2017-17', 'CC-MAIN-2017-13', 'CC-MAIN-2017-09', 'CC-MAIN-2017-04', 
              'CC-MAIN-2016-50', 'CC-MAIN-2016-44', 'CC-MAIN-2016-40', 'CC-MAIN-2016-36', 
              'CC-MAIN-2016-30', 'CC-MAIN-2016-26', 'CC-MAIN-2016-22', 'CC-MAIN-2016-18', 
              'CC-MAIN-2016-07', 'CC-MAIN-2015-48', 'CC-MAIN-2015-40', 'CC-MAIN-2015-35', 
              'CC-MAIN-2015-32', 'CC-MAIN-2015-27', 'CC-MAIN-2015-22', 'CC-MAIN-2015-18', 
              'CC-MAIN-2015-14', 'CC-MAIN-2015-11', 'CC-MAIN-2015-06', 'CC-MAIN-2014-52', 
              'CC-MAIN-2014-49', 'CC-MAIN-2014-42', 'CC-MAIN-2014-41', 'CC-MAIN-2014-35', 
              'CC-MAIN-2014-23', 'CC-MAIN-2014-15', 'CC-MAIN-2014-10', 'CC-MAIN-2013-48', 
              'CC-MAIN-2013-20', 'CC-MAIN-2012', 'CC-MAIN-2009-2010', 'CC-MAIN-2008-2009']

    blacklist = ['*/sports/*', '*rus.reuters*', '*fr.reuters*', '*br.reuters*', '*de.reuters*', '*es.reuters*', \
                 '*lta.reuters*', '*ara.reuters*', '*it.reuters*', '*ar.reuters*', '*blogs.reuters*', '*graphics.reuters*', \
                 '*jp.mobile.reuters*', '*live.special.reuters*', '*plus.reuters*', '*af.reuters*', \
                 '*in.reuters*', '*ru.reuters*', '*jp.reuters*', '*live.jp.reuters*', '*in.mobile.reuters*', \
                 '*br.mobile.reuters*', '*mx.reuters*', '*live.reuters*', '*cn.reuters*', '*agency.reuters*', \
                 '*widerimage.reuters*']

    for crawl in crawls:
        log(f"Starting pull for crawl: {crawl}")

        cc_source = sources.CommonCrawl()
        cc_source.set_crawls(crawl)
        collector = collect.Collector(cc_source)
        collector.only_political_articles() \
                 .on_sites(['reuters.com']) \
                 .exclude_in_url(blacklist) \
                 .in_language(lang='eng') \
                 .distinct()

        runner = run.Runner(collector, master_url=master_url, num_executors=5, executor_cores=8, executor_memory='64g', spark_conf=spark_conf)
        runner.send_to_database(db_connection_string, db_name, db_table)
        log(f"Finished pull for crawl: {crawl}")

if __name__ == '__main__':
    main()