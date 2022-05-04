import os

from seldonite import collect, sources, run

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
    db_table_in = 'reuters_news'

    mongo_source = sources.news.MongoDB(db_connection_string, db_name, db_table_in, partition_size_mb=32)

    # ignores georgia because is a US state and therefore creates lots of false positives
    collector = collect.Collector(mongo_source) \
        .by_keywords([
            'brent', 'crude', 'oil', 'cocoa', 'coffee', 'copper', 'corn',
            'cotton', 'cattle', 'gold', 'heating', 'hogs', 'cattle', 'lumber', 
            'gas', 'oat', 'palladium', 'platinum', 'rbob', 
            'gasoline', 'silver', 'soybean', 'meal'
        ])
    
    runner = run.Runner(collector, driver_cores=24, driver_memory='64g', python_executable='/home/ndg/users/bsteel2/miniconda3/envs/seldonite/bin/python')
    #runner = run.Runner(collector, master_url=master_url, num_executors=11, executor_cores=4, executor_memory='48g', driver_memory='64g', spark_conf=spark_conf)
    commodities_articles = runner.to_pandas()

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir_path, '..', '..', 'data', 'commodities')

    if not os.path.exists(data_path):
        os.mkdir(data_path)
    
    out_path = os.path.join(data_path, 'all_commodities.csv')
    commodities_articles.to_csv(out_path)

if __name__ == '__main__':
    main()