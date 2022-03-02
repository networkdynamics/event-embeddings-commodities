import os

from seldonite import analyze, collect, sources, run

def main():
    master_url = 'k8s://https://10.140.16.25:6443'
    db_connection_string = os.environ['MONGO_CONNECTION_STRING']

    spark_conf = {
        'spark.kubernetes.authenticate.driver.serviceAccountName': 'ben-dev',
        'spark.kubernetes.driver.pod.name': 'seldonite-driver-2',
        'spark.driver.host': 'seldonite-driver-2',
        'spark.driver.port': '7080'
    }

    db_name = 'political_events'
    db_table = 'reuters_news'

    mongo_source = sources.MongoDB(db_connection_string, db_name, db_table)
    collector = collect.Collector(mongo_source)
    analysis = analyze.Analyze(collector) \
        .article_domains()
    runner = run.Runner(analysis, master_url=master_url, executor_cores=8, executor_memory='64g', spark_conf=spark_conf)
    df = runner.to_pandas()

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir_path, '..', 'data')

    if not os.path.exists(data_path):
        os.mkdir(data_path)
    
    out_path = os.path.join(data_path, 'distinct_domains.csv')

    df.to_csv(out_path)

if __name__ == '__main__':
    main()