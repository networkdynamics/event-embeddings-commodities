import os

from seldonite import collect, sources, run

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
    db_table_in = 'reuters_news'
    db_table_out = 'reuters_news_2'

    mongo_source = sources.news.MongoDB(db_connection_string, db_name, db_table_in, partition_size_mb=64)

    collector = collect.Collector(mongo_source)
    #collector.distinct()
    collector.in_language(lang='en')

    runner = run.Runner(collector, master_url=master_url, num_executors=11, executor_cores=4, executor_memory='48g', driver_memory='64g', spark_conf=spark_conf)
    runner.send_to_database(db_connection_string, db_name, db_table_out)

if __name__ == '__main__':
    main()