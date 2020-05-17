import pandas as pd
import os, json, pyspark, sys, re, string
import numpy as np
assert sys.version_info >= (3, 5)
from pyspark.sql import SparkSession, functions, types
from pyspark.ml.clustering import LDA, LocalLDAModel
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel
def main():
    review_topics = spark.read.parquet("topic_modelling/review_topics_pos")
    cv_model = CountVectorizerModel.load("topic_modelling/cvmodel_pos")
    ldamodel = LocalLDAModel.load("topic_modelling/ldamodel_pos")
    f1out = open("topic_modelling/postive_topics","w+")
    topics = ldamodel.describeTopics(maxTermsPerTopic = 10).rdd.map(lambda x: list(x)).collect()
    vocabulary = cv_model.vocabulary
    for topic in range(len(topics)):
        towrite = "topic {} : \n".format(topic)
        f1out.write(towrite)
        words = topics[topic][1]
        scores = topics[topic][2]
        stri = ''
        for word in range(len(words)):
            stri += str(scores[word])+"*"+vocabulary[words[word]]+" + "
        f1out.write(stri[:-3]+"\n")
    f1out.close()


    review_topics = spark.read.parquet("topic_modelling/review_topics_neg")
    cv_model = CountVectorizerModel.load("topic_modelling/cvmodel_neg")
    ldamodel = LocalLDAModel.load("topic_modelling/ldamodel_neg")
    f2out = open("topic_modelling/negative_topics","w+")
    topics = ldamodel.describeTopics(maxTermsPerTopic = 10).rdd.map(lambda x: list(x)).collect()
    vocabulary = cv_model.vocabulary
    for topic in range(len(topics)):
        towrite = "topic {} : \n".format(topic)
        f2out.write(towrite)
        words = topics[topic][1]
        scores = topics[topic][2]
        stri = ''
        for word in range(len(words)):
            stri += str(scores[word])+"*"+vocabulary[words[word]]+" + "
        f2out.write(stri[:-3]+"\n")
    f2out.close()

if __name__ == '__main__':
    spark = SparkSession.builder.appName('reddit average').getOrCreate()
    assert spark.version >= '2.3'
    main()
