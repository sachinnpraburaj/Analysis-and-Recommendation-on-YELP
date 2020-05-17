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
from pyspark.sql.functions import lit, udf

def drop_from_array_(arr, item):
    return [x for x in arr if x != item]

drop_from_array = udf(drop_from_array_, types.ArrayType(types.StringType()))

def main(input):
    user = spark.read.json(input).repartition(80)
    split_col = functions.split(user['elite'],",")
    user = user.withColumn('elite_years',split_col).drop('elite').withColumn('elite',drop_from_array("elite_years", lit("None")))
    split_col = functions.split(user['friends'],",")
    user = user.withColumn('no_of_friends',split_col).drop('friends')
    user.createOrReplaceTempView("user")
    u_etl = spark.sql("SELECT user_id, review_count, name, size(no_of_friends) as friends, DATEDIFF(current_date(),yelping_since) as yelping_since, fans, size(elite) as elite, (compliment_writer + compliment_profile + compliment_plain + compliment_photos + compliment_note + compliment_more + compliment_list + compliment_hot + compliment_funny + compliment_funny + compliment_cute + compliment_cool) AS total_compliments, average_stars FROM user")
    u_etl.createOrReplaceTempView("u_etl")
    tot_comp = spark.sql("SELECT * from u_etl ORDER BY total_compliments DESC, review_count DESC")
    tot_comp.createOrReplaceTempView("user")

    review = spark.read.json("yelp-dataset/yelp_academic_dataset_review.json")
    business = spark.read.parquet("yelp-etl/business_etl")
    business.createOrReplaceTempView("business")
    review.createOrReplaceTempView("review")

    joined = spark.sql("SELECT u.user_id, AVG(b.latitude) as latitude, AVG(b.longitude) as longitude FROM user u INNER JOIN review r ON u.user_id = r.user_id INNER JOIN business b ON b.business_id = r.business_id GROUP BY u.user_id")
    joined.createOrReplaceTempView("joined")

    user_loc = spark.sql("SELECT u.*, j.latitude, j.longitude FROM user u INNER JOIN joined j ON u.user_id = j.user_id ORDER BY u.total_compliments DESC")

    user_revsince = user_loc.withColumn("revbysince",user_loc['review_count'] / user_loc['yelping_since']).select('user_id','name','average_stars', 'elite', 'fans', 'friends', 'review_count', 'total_compliments', 'yelping_since','revbysince','latitude','longitude').cache()
    
    col_names = ['average_stars', 'elite', 'fans', 'friends', 'review_count', 'total_compliments', 'yelping_since','revbysince']
    min = user_revsince.groupBy().min()
    max = user_revsince.groupBy().max()
    min_val = min.collect()
    max_val = max.collect()
    for i in range(len(col_names)):
        user_revsince = user_revsince.withColumn("n_"+col_names[i],(user_revsince[col_names[i]] - min_val[0][i])/(max_val[0][i] - min_val[0][i]))
    
    user_val = user_revsince.withColumn('score',(6*user_revsince['n_average_stars']+24*user_revsince['n_elite']+12*(user_revsince['n_fans']+user_revsince['n_friends'])+30*user_revsince['n_total_compliments']+16*user_revsince['n_revbysince'])/100)
    
    max = user_val.groupBy().max()
    max_val = max.collect()
    user_val = user_val.withColumn('validation',(user_val['score']/max_val[0][18])*100)
    
    user_out = user_val.select(user_val['user_id'],user_val['name'],user_val['average_stars'],user_val['elite'],user_val['fans'],user_val['friends'],user_val['total_compliments'],user_val['review_count'],user_val['yelping_since'],user_val['validation'],user_val['latitude'],user_val['longitude'])


    user_out.write.parquet("yelp-etl/user_etl", mode="overwrite")
    
    

if __name__ == '__main__':
    data_path = os.getcwd()+"/yelp-dataset/"
    user_filepath = data_path + 'yelp_academic_dataset_user.json'
    sc = sc(appName="Yelp")
    sqlContext = SQLContext(sc)
    spark = SparkSession.builder.appName('reddit average').getOrCreate()
    assert spark.version >= '2.3'
    main(user_filepath)
