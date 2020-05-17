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

def attribute_score(attribute):
    att = spark.sql("SELECT attributes.{attr} as {attr}, category, stars FROM for_att".format(attr=attribute)).dropna()
    att.createOrReplaceTempView("att")

    att_group = spark.sql("SELECT category, {attr}, AVG(stars) AS stars FROM att GROUP BY category, {attr} ORDER BY stars".format(attr=attribute))
    att_group.coalesce(1).write.json("analysis/{attr}".format(attr=attribute), mode = "overwrite")

def main(input):
    business = spark.read.parquet(input).cache()
    business.createOrReplaceTempView("business")

    ## Average review count and stars by city and category
    for_avg = spark.sql("SELECT state, city, stars, review_count, explode(categories) AS category FROM business ").cache()
    for_avg.createOrReplaceTempView('for_avg')

    avg_city = spark.sql("SELECT city, category, AVG(review_count)as avg_review_count, AVG(stars) as avg_stars FROM for_avg GROUP BY city, category ORDER BY city, avg_review_count DESC")
    avg_city.coalesce(1).write.json("analysis/average_stars_city", mode = "overwrite")

    ## Average review count and stars by state and category
    avg_state = spark.sql("SELECT state, category, AVG(review_count)as avg_review_count, AVG(stars) as avg_stars FROM for_avg GROUP BY state, category ORDER BY state, avg_review_count DESC")
    avg_state.coalesce(1).write.json("analysis/average_stars_state", mode = "overwrite")

    ## Data based on Attribute
    for_att = spark.sql("SELECT attributes, stars, explode(categories) AS category FROM business")
    for_att.createOrReplaceTempView("for_att")
    attribute = 'RestaurantsTakeout'
    attribute_score(attribute)

    ## Average stars for open and closed businesses
    open_close = spark.sql("SELECT is_open, AVG(stars) AS avg_stars, AVG(review_count) as avg_review FROM business GROUP BY is_open")
    open_close.coalesce(1).write.json("analysis/open_close", mode = "overwrite")


    ## Top 15 business categories
    top_cat = spark.sql("SELECT category, COUNT(*) as freq FROM for_avg GROUP BY category ORDER BY freq DESC")
    top_cat.coalesce(1).write.json("analysis/top_category", mode = "overwrite")

    ## Top 15 business categories - in every city
    top_cat_city = spark.sql("SELECT city, category, COUNT(*) as freq FROM for_avg GROUP BY city, category ORDER BY city, freq DESC")
    top_cat_city.coalesce(1).write.json("analysis/top_category_city", mode = "overwrite")


    ## Cities with most businesses
    bus_city = spark.sql("SELECT city, COUNT(business_id) as no_of_bus FROM business GROUP BY city ORDER BY no_of_bus DESC")
    bus_city.coalesce(1).write.json("analysis/top_business_city", mode = "overwrite")

    review = spark.read.parquet("yelp-etl/review_etl").cache()
    review.createOrReplaceTempView("review")

    ## Top-businesses based on 5-star ratings
    top_bus_rat = spark.sql("SELECT b.business_id, b.name, count(r.stars) AS tot_5star FROM business b INNER JOIN review r ON b.business_id = r.business_id WHERE r.stars = 5 GROUP BY b.business_id, b.name ORDER BY tot_5star DESC")
    top_bus_rat.coalesce(1).write.json("analysis/top_5star_bus", mode = "overwrite")


if __name__ == '__main__':
    data_path = os.getcwd()+"/yelp-etl/"
    Business_filepath = data_path + 'business_etl'
    sc = sc(appName="Yelp")
    sqlContext = SQLContext(sc)
    spark = SparkSession.builder.appName('reddit average').getOrCreate()
    assert spark.version >= '2.3'
    main(Business_filepath)
