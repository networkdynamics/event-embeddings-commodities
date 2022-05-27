import os

from seldonite import collect, embed, nlp, sources, run

def main():

    local_spark_conf = {
        'spark.driver.maxResultSize': '16g'
    }

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir_path, '..', '..', 'data')

    if not os.path.exists(data_path):
        os.mkdir(data_path)
    
    csv_path = os.path.join(data_path, 'relations', 'all_relations.csv')
    
    #countries = ['Afghanistan', 'United Kingdom', 'Canada', 'China', 'Cuba', 'France', 'Germany', 'Iran', 'Iraq', 'Japan', 'Mexico', 'Russia', 'Spain', 'United States', 'Vietnam']
    countries = ['United Kingdom', 'Canada', 'China', 'Cuba', 'France', 'Germany', 'Iran', 'Iraq', 'Japan', 'Mexico', 'Russia', 'Spain', 'United States', 'Vietnam']

    spark_manager = None
    for country in countries:
        source = sources.news.CSV(csv_path)

        ignore_countries = ['Georgia'] if country == 'United States' else []
        collector = collect.Collector(source) \
            .mentions_countries(countries=[country], min_num_countries=2, ignore_countries=ignore_countries, output=True) \
            .in_language(lang='en')
        
        runner = run.Runner(collector, driver_cores=24, driver_memory='64g', python_executable='/home/ndg/users/bsteel2/.conda/envs/seldonite/bin/python', keep_alive=True, spark_conf=local_spark_conf)
        #runner = run.Runner(collector, master_url=master_url, num_executors=11, executor_cores=4, executor_memory='48g', driver_memory='64g', spark_conf=spark_conf)
        
        if spark_manager:
            runner.set_spark_manager(spark_manager)
        commodities_articles = runner.to_pandas()
        spark_manager = runner.get_spark_manager()

        out_path = os.path.join(data_path, 'relations', f"{country.lower().replace(' ', '_')}_articles.csv")
        commodities_articles.to_csv(out_path)

    spark_manager.stop()

if __name__ == '__main__':
    main()