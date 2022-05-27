import datetime
import os
import time

from seldonite import sources, collect, nlp, graphs, run

def main():
    master_url = 'k8s://https://10.140.16.25:6443'
    db_connection_string = os.environ['MONGO_CONNECTION_STRING']

    spark_conf = {
        'spark.kubernetes.authenticate.driver.serviceAccountName': 'ben-dev',
        'spark.kubernetes.driver.pod.name': 'seldonite-driver',
        'spark.driver.host': 'seldonite-driver',
        'spark.driver.port': '7078',
        'spark.kubernetes.executor.volumes.hostPath.fake-nfs-mount.mount.path': '/root',
        'spark.kubernetes.executor.volumes.hostPath.fake-nfs-mount.options.path': '/var/nfs/spark'
    }
        #'spark.kubernetes.executor.volumes.nfs.shared-nfs-mount.mount.path': '/root',
        #'spark.kubernetes.executor.volumes.nfs.shared-nfs-mount.options.server': '10.100.90.152',
        #'spark.kubernetes.executor.volumes.nfs.shared-nfs-mount.options.path': '/var/nfs/spark'
    #}

    db_name = 'political_events'
    db_table_in = 'reuters_news'

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir_path, '..', '..', 'data', 'better_news_graphs')

    if not os.path.exists(data_path):
        os.mkdir(data_path)

    tfidf_path = os.path.join('/root', 'tfidf', 'all_articles')

    spark_manager = None

    first_year = 2006
    last_year = 2022
    for year in range(first_year, last_year):
        for half in range(2):
            print(f"Starting year {year} half {half}")
            mongo_source = sources.news.MongoDB(db_connection_string, db_name, db_table_in, partition_size_mb=16)
            #mongo_source = sources.CSV('/root/google_protest_news.csv')
            
            if half == 0:
                start_date = datetime.date(year, 1, 1)
                end_date = datetime.date(year, 6, 1) - datetime.timedelta(days=1)
            elif half == 1:
                start_date = datetime.date(year, 6, 1)
                end_date = datetime.date(year + 1, 1, 1) - datetime.timedelta(days=1)
            
            collector = collect.Collector(mongo_source) \
                .in_date_range(start_date, end_date) \
                .sample(50000)

            nl_processor = nlp.NLP(collector)
            nl_processor.top_tfidf(20, load_path=tfidf_path)

            graph_constructor = graphs.Graph(nl_processor)
            graph_constructor.build_tfidf_graph()

            runner = run.Runner(graph_constructor, master_url=master_url, num_executors=11, executor_cores=4, executor_memory='48g', driver_memory='64g', spark_conf=spark_conf, keep_alive=True)
            #runner = run.Runner(graph_constructor, driver_cores=24, driver_memory='64g', python_executable='/home/ndg/users/bsteel2/miniconda3/envs/seldonite/bin/python')
            if spark_manager:
                runner.set_spark_manager(spark_manager)
            article_df, words_df, edges_df = runner.to_pandas()
            spark_manager = runner.get_spark_manager()
            
            article_path = os.path.join(data_path, f"{str(year)[2:]}_{half}_articles.csv")
            words_path = os.path.join(data_path, f"{str(year)[2:]}_{half}_words.csv")
            edges_path = os.path.join(data_path, f"{str(year)[2:]}_{half}_edges.csv")

            article_df.to_csv(article_path)
            words_df.to_csv(words_path)
            edges_df.to_csv(edges_path)

            print(f"Finished year {year} half {half}")

if __name__ == '__main__':
    main()