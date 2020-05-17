from pyspark import SparkContext as sc
import pandas as pd
import os
import numpy as np
from pyspark.sql import SQLContext
import json
import pyspark
import sys
assert sys.version_info >= (3, 5)
from pyspark.sql import SparkSession, functions, types


def main(input):
    business = spark.read.json(input).repartition(80)
    split_col = functions.split(business['categories'], ',')
    business = business.withColumn("categories",split_col).filter(business["city"] != "").dropna()
    business.createOrReplaceTempView("business")

    b_etl = spark.sql("SELECT business_id, name, city, state, latitude, longitude, stars, review_count, is_open, categories, attributes FROM business").cache()
    b_etl.createOrReplaceTempView("b_etl")
    outlier = spark.sql("SELECT b1.business_id, SQRT(POWER(b1.latitude - b2.avg_lat, 2) + POWER(b1.longitude - b2.avg_long, 2)) as dist FROM b_etl b1 INNER JOIN (SELECT state, AVG(latitude) as avg_lat, AVG(longitude) as avg_long FROM b_etl GROUP BY state) b2 ON b1.state = b2.state ORDER BY dist DESC")
    outlier.createOrReplaceTempView("outlier")
    joined = spark.sql("SELECT b.* FROM b_etl b INNER JOIN outlier o ON b.business_id = o.business_id WHERE o.dist<10")
    joined.write.parquet("yelp-etl/business_etl", mode = "overwrite")




if __name__ == '__main__':
    data_path = os.getcwd()+"/yelp-dataset/"
    Business_filepath = data_path + 'yelp_academic_dataset_business.json'
    sc = sc(appName="Yelp")
    sqlContext = SQLContext(sc)
    spark = SparkSession.builder.appName('reddit average').getOrCreate()
    assert spark.version >= '2.3'
    main(Business_filepath)
