import datetime
import os
import re

import pyspark.sql as psql

from seldonite import collect, sources, run

def clean_date(txt):
    try:
        if txt.startswith('java.util.GregorianCalendar'):
            year_match = re.search('(?<=,YEAR=)\d+', txt)
            month_match = re.search('(?<=,MONTH=)\d+', txt)
            day_match = re.search('(?<=,DAY_OF_MONTH=)\d+', txt)

            if (year_match and month_match and day_match):
                year = int(year_match.group(0))
                month = int(month_match.group(0)) + 1
                day = int(day_match.group(0))
                return datetime.date(year, month, day)
            else:
                raise Exception("Unrecognised date format")
        elif txt.startswith('{"$date":'):
            match = re.search('[0-9]+', txt)
            if match:
                return datetime.date.fromtimestamp(int(match.group(0)[:-3]))
            else:
                raise Exception("Unrecognised date format")
        else:
            raise Exception("Unrecognised date format")
    except Exception as e:
        raise Exception(f"Received error: {str(e)}, for text: {txt}")

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

    clean_date_udf = psql.functions.udf(clean_date, returnType=psql.types.DateType())
    collector.apply_udf(clean_date_udf, 'publish_date')

    runner = run.Runner(collector, master_url=master_url, num_executors=11, executor_cores=4, executor_memory='48g', driver_memory='64g', spark_conf=spark_conf)
    runner.send_to_database(db_connection_string, db_name, db_table_out)

if __name__ == '__main__':
    main()