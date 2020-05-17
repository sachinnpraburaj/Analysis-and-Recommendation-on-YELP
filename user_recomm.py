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

def main(input,uid):
    user = spark.read.parquet(input)
    review = spark.read.parquet("yelp-etl/review_etl")
    business = spark.read.parquet("yelp-etl/business_etl")

    user.createOrReplaceTempView("user")
    business.createOrReplaceTempView("business")
    review.createOrReplaceTempView("review")


    # Location based recommendations
    user_bus_loc = spark.sql("SELECT b.business_id, b.name, SQRT(POWER(b.latitude - u.latitude, 2) + POWER(b.longitude - u.longitude, 2)) as dist FROM user u CROSS JOIN business b WHERE u.user_id = "+uid+" ORDER BY dist")
    # user_bus_loc.createOrReplaceTempView("user_bus_loc")

    # Join tables based on users availed services
    joined = spark.sql("SELECT u.user_id, b.business_id, b.categories, b.name, r.stars, r.label FROM user u INNER JOIN review r ON u.user_id = r.user_id INNER JOIN business b ON r.business_id = b.business_id WHERE u.user_id = "+uid).cache()
    joined.createOrReplaceTempView("joined")

    # Category based recommendations
    user_cat = spark.sql("SELECT business_id, name, explode(categories) as category FROM joined")
    user_cat.createOrReplaceTempView("user_cat")

    freq_cat = spark.sql("SELECT category, COUNT(*) as cat_count FROM user_cat GROUP BY category ORDER BY cat_count DESC")
    freq_cat.createOrReplaceTempView("freq_cat")

    user_bus_cat = spark.sql("SELECT business_id, name, COUNT(*) AS no_of_cat FROM (SELECT u.business_id, u.name FROM user_cat u INNER JOIN freq_cat f on u.category = f.category) GROUP BY business_id, name ORDER BY no_of_cat desc")
    # user_bus_cat.createOrReplaceTempView("user_bus_cat")

    # Ratings based recommendations
    user_bus_rat = spark.sql("SELECT business_id, name, AVG(stars) as avg_rating FROM joined GROUP BY business_id, name ORDER BY avg_rating DESC")
    # user_bus_rat.createOrReplaceTempView("user_bus_rat")

    # Review based recommendations
    user_bus_rev = spark.sql("SELECT business_id, name, label FROM joined ORDER BY label DESC")
    user_bus_rev = user_bus_rev.withColumn("review", functions.when(user_bus_rev["label"] == 0,-1).otherwise(1)).drop('label')
    # user_bus_rev.createOrReplaceTempView("user_bus_rev")

    # Overall recommendations
    user_rec = user_bus_loc.join(user_bus_cat.drop('name'),"business_id","inner").join(user_bus_rat.drop('name'),"business_id","outer").join(user_bus_rev.drop('name'),"business_id","outer").cache()


    # Normalize
    col_names = ["dist","no_of_cat","avg_rating"]
    min = user_rec.groupBy().min().collect()
    max = user_rec.groupBy().max().collect()
    for i in range(len(col_names)):
        user_rec = user_rec.withColumn("n_"+col_names[i], (user_rec[col_names[i]] - min[0][i]) / (max[0][i] - min[0][i]) )
    user_rec.createOrReplaceTempView("user_rec")

    rec_score = spark.sql("SELECT business_id, name, dist, no_of_cat, avg_rating, review, (n_no_of_cat+2*(n_avg_rating-n_dist)+review) AS rec_score FROM user_rec ORDER BY rec_score DESC")


    # user_bus_cat.limit(15).show()
    # user_bus_loc.limit(15).show()
    # rec_score.show()
    user_bus_cat.coalesce(1).write.json("analysis/user_cat_recc",mode="overwrite")
    user_bus_loc.coalesce(1).write.json("analysis/user_loc_recc",mode="overwrite")
    rec_score.coalesce(1).write.json("analysis/user_recc",mode="overwrite")

# "'CxDOIDnH8gp9KXzpBHJYXw'"
# "'Akt0llUBaVa1Qxi8Ogdv4Q'"
# "'8k3aO-mPeyhbR5HUucA5aA'"
# "'Tqm7Wu7IBJ1td3Ab5ZpUhw'"
if __name__ == '__main__':
    data_path = os.getcwd()+"/yelp-etl/"
    user_filepath = data_path + 'user_etl'
    sc = sc(appName="Yelp")
    sqlContext = SQLContext(sc)
    spark = SparkSession.builder.appName('reddit average').getOrCreate()
    assert spark.version >= '2.3'
    user_id = sys.argv[1]
    main(user_filepath,user_id)
