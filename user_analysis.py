import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from pyspark.sql import SparkSession, functions, types
from pyspark.ml.feature import StringIndexer, VectorAssembler, SQLTransformer
spark = SparkSession.builder.appName('user analysis').getOrCreate()
assert spark.version >= '2.3'

def main():
    # main logic starts here
    user = spark.read.parquet("yelp-etl/user_etl")
    user.createOrReplaceTempView("user")

    review = spark.read.parquet("yelp-etl/review_etl")
    review.createOrReplaceTempView("review")

    business = spark.read.parquet("yelp-etl/business_etl")
    business.createOrReplaceTempView("business")

    ## most availed category of business by a user
    bus = spark.sql("SELECT u.user_id,b.stars,EXPLODE(b.categories) AS categories FROM user u INNER JOIN review r ON u.user_id = r.user_id INNER JOIN business b ON r.business_id = b.business_id ORDER BY u.user_id")
    bus.createOrReplaceTempView("bus")

    bus_val = spark.sql("SELECT user_id,categories,COUNT(*) AS times_visited FROM bus GROUP BY user_id,categories ORDER BY user_id,times_visited DESC")
    bus_val.coalesce(1).write.json("analysis/user_freq_cat", mode="overwrite")

    ##  average stars given by user for each category
    avg_stars = spark.sql("SELECT user_id,categories,AVG(stars) AS avg_stars from bus GROUP BY user_id,categories")
    avg_stars.coalesce(1).write.json("analysis/user_cat_stars", mode="overwrite")

    ## number of positive and negative reviews given by a user
    user_reviews = spark.sql("SELECT r.label, COUNT(*) as num_rev FROM user u INNER JOIN review r ON u.user_id = r.user_id WHERE u.user_id = 'CxDOIDnH8gp9KXzpBHJYXw' GROUP BY r.label")
    user_reviews.coalesce(1).write.json("analysis/user_num_rev",mode="overwrite")

if __name__ == '__main__':
    main()
