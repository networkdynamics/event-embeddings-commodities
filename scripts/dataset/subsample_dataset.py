import datetime
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
    db_table_out = 'reuters_news_reduced'

    start_year = 2007
    start_date = datetime.date(start_year, 1, 1)
    end_date = datetime.date(2021, 1, 1)

    num_years = end_date.year - start_date.year + 1
    year_intervals = [(datetime.date(start_year + i, 1, 1), datetime.date(start_year + i + 1, 1, 1) - datetime.timedelta(days=1)) for i in range(num_years)]
    #counts = [1000000 // num_months for _ in range(num_months)]

    counts = [18000, 40000, 42000, 44000, 44000, 54000, 57000, 78000, 93000, 100000, 100000, 100000, 100000, 100000, 45000]

    for year_idx in range(num_years):

        year_start = year_intervals[year_idx][0]
        year_end = year_intervals[year_idx][1]
        num_in_year = counts[year_idx]

        mongo_source = sources.MongoDB(db_connection_string, db_name, db_table_in, partition_size_mb=16)

        collector = collect.Collector(mongo_source)
        collector.in_date_range(year_start, year_end) \
                 .sample(num_in_year)

        runner = run.Runner(collector, master_url=master_url, num_executors=2, executor_cores=22, executor_memory='420g', spark_conf=spark_conf)
        runner.send_to_database(db_connection_string, db_name, db_table_out)

if __name__ == '__main__':
    main()