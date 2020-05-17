import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types
from pyspark.ml.feature import StringIndexer, VectorAssembler, SQLTransformer
spark = SparkSession.builder.appName('valid reviews').getOrCreate()
assert spark.version >= '2.3'

def main():
    user = spark.read.parquet("yelp-etl/user_etl")
    user.createOrReplaceTempView("user")

    review = spark.read.parquet("yelp-etl/review_etl")
    review.createOrReplaceTempView("review")

    business = spark.read.parquet("yelp-etl/business_etl")
    business.createOrReplaceTempView("business")

    ## business with maximum reviews
    bus = spark.sql("SELECT bu.* FROM (SELECT b.business_id, COUNT(*) as rev_count FROM review r INNER JOIN business b ON r.business_id = b.business_id GROUP BY b.business_id ORDER BY rev_count DESC LIMIT 1) t INNER JOIN business bu ON bu.business_id = t.business_id")
    bus.createOrReplaceTempView("bus")

    ## 10 positive and negative reviews for a business based on user validation
    combine = spark.sql("SELECT b.business_id,r.review_id,r.text,r.label,u.user_id,u.name,u.validation FROM bus b INNER JOIN review r ON b.business_id = r.business_id INNER JOIN user u ON r.user_id = u.user_id").cache()
    combine.createOrReplaceTempView("combine")

    file = open("analysis/top_pos_neg_rev","w+")

    pos = spark.sql("SELECT label,text,validation FROM combine WHERE label == 1 ORDER BY validation DESC LIMIT 10")

    pos_rev = pos.rdd.map(lambda x:list(x)).collect()

    for i in range(len(pos_rev)):
        stri = "class:"+str(pos_rev[i][0])+"|review:"+pos_rev[i][1].strip("\n")+"|validation_score:"+str(pos_rev[i][2])+"\n*************\n"
        file.write(stri)

    neg = spark.sql("SELECT label,text,validation FROM combine WHERE label == 0 ORDER BY validation DESC LIMIT 10")

    neg_rev = neg.rdd.map(lambda x:list(x)).collect()

    for i in range(len(neg_rev)):
        stri = "class:"+str(neg_rev[i][0])+"|review:"+neg_rev[i][1].strip("\n")+"|validation_score:"+str(neg_rev[i][2])+"\n*************\n"
        file.write(stri)

    file.close()

if __name__ == '__main__':
    main()
