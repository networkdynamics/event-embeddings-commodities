import os

from seldonite import collect, embed, nlp, sources, run

def main():

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir_path, '..', '..', 'data')

    if not os.path.exists(data_path):
        os.mkdir(data_path)
    
    csv_path = os.path.join(data_path, 'commodity_data', 'all_commodities.csv')
    
    commodities = {
        'brent_crude_oil': ['brent', 'crude', 'oil'],
        'crude_oil': ['crude', 'oil'],
        'natural_gas': ['gas'],
        'rbob_gasoline': ['rbob', 'gasoline'],
        'copper': ['copper'],
        'palladium': ['palladium'],
        'platinum': ['platinum'],
        'gold': ['gold'],
        'silver': ['silver'],
        'corn': ['corn'],
        'oat': ['oat', 'oats'],
        'cotton': ['cotton'],
        'lumber': ['lumber'],
        'cocoa': ['cocoa'],
        'coffee': ['coffee'],
        'feeder_cattle': ['cattle'],
        'heating_oil': ['heating', 'oil'],
        'lean_hogs': ['hogs'],
        'live_cattle': ['cattle'],
        'soybean_meal': ['soybean', 'soybeans'],
        'soybean_oil': ['soybean', 'soybeans'],
        'soybean': ['soybean', 'soybeans'],
        'sugar': ['sugar'],
        'wheat': ['wheat']
    }

    spark_manager = None
    for commodity, commodity_key_words in commodities.items():
        source = sources.news.CSV(csv_path)

        collector = collect.Collector(source) \
            .by_keywords(commodity_key_words) \
            .in_language(lang='en')
        
        runner = run.Runner(collector, driver_cores=24, driver_memory='64g', python_executable='/home/ndg/users/bsteel2/.conda/envs/seldonite/bin/python', keep_alive=True)
        #runner = run.Runner(collector, master_url=master_url, num_executors=11, executor_cores=4, executor_memory='48g', driver_memory='64g', spark_conf=spark_conf)
        
        if spark_manager:
            runner.set_spark_manager(spark_manager)
        commodities_articles = runner.to_pandas()
        spark_manager = runner.get_spark_manager()

        out_path = os.path.join(data_path, 'commodity_data', f"{commodity}_articles.csv")
        commodities_articles.to_csv(out_path)

    spark_manager.stop()

if __name__ == '__main__':
    main()