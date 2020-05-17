import sys, json, string, re
from pyspark import SparkConf, SparkContext
from pyspark.sql.types import *
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from pyspark.ml import *
from pyspark.ml.feature import *
from pyspark.sql.functions import udf
from pyspark.sql import SparkSession, functions, types
from pyspark.ml.classification import *
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import *
from pyspark.mllib.linalg import Vectors as MLLibVectors
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import numpy as np

spark = SparkSession.builder.appName('review class').getOrCreate()
assert spark.version >= '2.3'

inputs = "yelp-dataset/yelp_academic_dataset_review.json"
model_file = "analysis/review_class_model"
review = spark.read.json(inputs).repartition(400).limit(1000000)


def remove_punct(text):
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    nopunct = regex.sub(" ", text)
    return nopunct

def convert_rating(rating):
    if rating >= 3:
        return 1
    elif rating<3:
        return 0

punct_remover = udf(lambda x: remove_punct(x))
rating_convert = udf(lambda x: convert_rating(x))

review_df = review.select('review_id', punct_remover('text'), rating_convert('stars'))
review_df = review_df.withColumnRenamed('<lambda>(text)', 'text').withColumn('label', review_df["<lambda>(stars)"].cast(IntegerType())).drop('<lambda>(stars)')

tok = Tokenizer(inputCol="text", outputCol="words")
stopword_rm = StopWordsRemover(inputCol='words', outputCol='words_nsw')
pipeline = Pipeline(stages=[tok, stopword_rm])
review_tokenized = pipeline.fit(review_df).transform(review_df)

n = 3

ngram = NGram(inputCol = 'words', outputCol = 'ngram', n = n)
add_ngram = ngram.transform(review_tokenized)

ngrams = add_ngram.rdd.flatMap(lambda x: x[-1]).filter(lambda x: len(x.split())==n)
ngram_tally = ngrams.map(lambda x: (x, 1)).reduceByKey(lambda x,y: x+y).sortBy(lambda x: x[1], ascending=False).filter(lambda x: x[1]>=20)
ngram_list = ngram_tally.map(lambda x: x[0]).collect()


def ngram_concat(text):
    text1 = text.lower()
    for ngram in ngram_list:
        if ngram in text1:
            new_ngram = ngram.replace(' ', '_')
            text1 = text1.replace(ngram, new_ngram)
    return text1

ngram_df = udf(lambda x: ngram_concat(x))
ngram_df = review_tokenized.select(ngram_df('text'), 'label', 'review_id').withColumnRenamed('<lambda>(text)','text')


tok = Tokenizer(inputCol="text", outputCol="words")
review_tokenized = tok.transform(review_df)
tokenized_ngram = tok.transform(ngram_df)
tokenized_ngram = stopword_rm.transform(tokenized_ngram)

stopword_rm = StopWordsRemover(inputCol='words', outputCol='words_nsw')
review_tokenized = stopword_rm.transform(review_tokenized)

cv = CountVectorizer(inputCol='words_nsw', outputCol='tf')
cvModel = cv.fit(review_tokenized)
count_vectorized = cvModel.transform(review_tokenized)

idf_ngram = IDF().setInputCol('tf').setOutputCol('tfidf')
tfidfModel_ngram = idf_ngram.fit(count_vectorized)
tfidf_df = tfidfModel_ngram.transform(count_vectorized)

splits = tfidf_df.select(['tfidf', 'label','review_id']).randomSplit([0.8,0.2],seed=100)
train = splits[0]
test = splits[1]

classifier = RandomForestClassifier(labelCol="label",featuresCol = "tfidf", numTrees=20)
model = classifier.fit(train)
model.write().overwrite().save(model_file)
prediction_test = model.transform(test)
prediction_train = model.transform(train)

output = prediction_train.union(prediction_test)
output.select('review_id','prediction','label').write.json("yelp-etl/review_class", mode="overwrite")
