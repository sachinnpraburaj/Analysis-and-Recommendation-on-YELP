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
from pyspark.sql.types import *
from pyspark.sql.functions import lit, udf

def main(inputs):
    review = spark.read.json(inputs).repartition(400)
    review.createOrReplaceTempView("review")
    r_trans = spark.sql("SELECT review_id, user_id, business_id, stars, text, (useful+funny+cool) AS votes FROM review")
    rclass = spark.read.json("yelp-etl/review_class").select("review_id","prediction").withColumnRenamed('prediction', 'label')
    joined = r_trans.join(rclass,"review_id")
    joined.write.parquet("yelp-etl/review_etl",mode = "overwrite")



if __name__ == '__main__':
    data_path = os.getcwd()+"/yelp-dataset/"
    user_filepath = data_path + 'yelp_academic_dataset_review.json'
    sc = sc(appName="Yelp")
    sqlContext = SQLContext(sc)
    spark = SparkSession.builder.appName('reddit average').getOrCreate()
    assert spark.version >= '2.3'
    main(user_filepath)
