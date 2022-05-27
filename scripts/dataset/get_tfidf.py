import os

from seldonite import collect, nlp, run, sources

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

    mongo_source = sources.news.MongoDB(db_connection_string, db_name, db_table_in, partition_size_mb=32)

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir_path, '..', '..', 'data', 'tfidf')

    if not os.path.exists(data_path):
        os.mkdir(data_path)
    
    out_path = os.path.join(data_path, 'all_articles')

    collector = collect.Collector(mongo_source)
    nl_processor = nlp.NLP(collector) \
        .top_tfidf(20, save_path=out_path)
    
    runner = run.Runner(nl_processor, driver_cores=24, driver_memory='64g', python_executable='/home/ndg/users/bsteel2/miniconda3/envs/seldonite/bin/python')
    #runner = run.Runner(nl_processor, master_url=master_url, num_executors=11, executor_cores=4, executor_memory='48g', driver_memory='64g', spark_conf=spark_conf)
    runner.run()


if __name__ == '__main__':
    main()