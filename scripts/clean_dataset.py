import os

from seldonite import collect, sources, run

def main():
    master_url = 'k8s://https://10.140.16.25:6443'
    db_connection_string = os.environ['MONGO_CONNECTION_STRING']

    spark_conf = {
        'spark.kubernetes.authenticate.driver.serviceAccountName': 'ben-dev',
        'spark.kubernetes.driver.pod.name': 'seldonite-driver',
        'spark.driver.host': 'seldonite-driver-2',
        'spark.driver.port': '7080'
    }

    db_name = 'political_events'
    db_table_in = 'reuters_news'
    db_table_out = 'reuters_news_2'

    mongo_source = sources.MongoDB(db_connection_string, db_name, db_table_in, partition_size_mb=64)

    blacklist = ['*/sports/*', '*rus.reuters*', '*fr.reuters*', '*br.reuters*', '*de.reuters*', '*es.reuters*', \
                 '*lta.reuters*', '*ara.reuters*', '*it.reuters*', '*ar.reuters*', '*blogs.reuters*', '*graphics.reuters*', \
                 '*jp.mobile.reuters*', '*live.special.reuters*', '*plus.reuters*', '*af.reuters*', \
                 '*in.reuters*', '*ru.reuters*', '*jp.reuters*', '*live.jp.reuters*', '*in.mobile.reuters*', \
                 '*br.mobile.reuters*', '*mx.reuters*', '*live.reuters*', '*cn.reuters*', '*agency.reuters*', \
                 '*widerimage.reuters*']

    collector = collect.Collector(mongo_source)
    collector.exclude_in_url(blacklist) \
             .distinct()

    runner = run.Runner(collector, master_url=master_url, num_executors=1, executor_cores=16, executor_memory='420g', spark_conf=spark_conf)
    runner.send_to_database(db_connection_string, db_name, db_table_out)

if __name__ == '__main__':
    main()