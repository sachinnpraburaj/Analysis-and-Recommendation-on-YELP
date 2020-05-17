import pandas as pd
import os, json, pyspark, sys, re, string
import numpy as np
assert sys.version_info >= (3, 5)
from pyspark.sql import SparkSession, functions, types
from pyspark.ml.feature import *
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from pyspark.ml.clustering import LDA
from pyspark.ml import Pipeline

# remove punctuation
def remove_punct(text):
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    rem_space = re.compile('\W+|\W+$|[^\w\s]+|_')
    nopunct = regex.sub(" ", text)
    toret = rem_space.sub(" ", nopunct)
    return toret

# lemmatization
def lemmatize(in_vec):
    out_vec = []
    for t in in_vec:
        t_lemm = WordNetLemmatizer().lemmatize(t, pos='v')
        if len(t_lemm) > 2:
            out_vec.append(t_lemm)
    return out_vec

# stemming
def stem(in_vec):
    out_vec = []
    for t in in_vec:
        t_stem = stemmer.stem(t)
        if len(t_stem) > 2:
            out_vec.append(t_stem)
    return out_vec

def main():
    spark.sql("CLEAR CACHE")
    business = spark.read.parquet("yelp-etl/business_etl").repartition(8)
    business.createOrReplaceTempView("business")
    review = spark.read.parquet("yelp-etl/review_etl").repartition(16)#.cache()
    review.createOrReplaceTempView("review")

    ## Location based reviews
    # spark.sql("SELECT b.state, COUNT(*) AS bus_rev_count FROM business b INNER JOIN review r ON b.business_id = r.business_id GROUP BY b.state ORDER BY bus_rev_count DESC").show()
    #
    # ## Choosing reviews from Pennsylvania (state = "PA")
    pa_bus_rev = spark.sql("SELECT r.review_id, b.business_id, r.text, r.label FROM business b INNER JOIN review r ON b.business_id = r.business_id WHERE b.state = 'PA' AND r.label = 1")

    ## Remove punctuations and spaces
    punct_remover = functions.udf(lambda x: remove_punct(x))
    review_df = pa_bus_rev.select('review_id', 'business_id', punct_remover('text')).withColumnRenamed('<lambda>(text)', 'text')

    ## Tokenize
    tok = Tokenizer(inputCol="text", outputCol="words")

    ## Remove stop words
    stopwordList = ['','i','get','got','also','really','would','one','good','like','great','tri','love','two','three','took','awesome','me','bad','horrible','disgusting','terrible','fabulous','amazing','terrific','worst','best','fine','excellent','acceptable','my','exceptional','satisfactory','satisfying','super','awful','atrocious','unacceptable','poor','sad','gross','authentic','myself','cheap','expensive','we','our','ours','ourselves','you','your','yours','yourself','yourselves', 'he', 'him', 'his', 'himself','she','her','hers','herself','it','its','itself','they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then','once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each','few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn','weren', 'won', 'wouldn']

    stopword_rm = StopWordsRemover(inputCol="words", outputCol="words_nsw", stopWords=stopwordList)

    pipestages = [tok,stopword_rm]
    pipeline = Pipeline(stages = pipestages)
    model = pipeline.fit(review_df)
    tokenized_df = model.transform(review_df)

    ## Lemmatizing
    lemmatize_udf = functions.udf(lambda x: lemmatize(x), types.ArrayType(types.StringType()))
    lemmatized_df = tokenized_df.withColumn("lemmatized",lemmatize_udf("words_nsw")).select("review_id","business_id","lemmatized")
    ## Stemming
    stemmer_udf = functions.udf(lambda x: stem(x), types.ArrayType(types.StringType()))
    stemmed_df = lemmatized_df.withColumn("stemmed", stemmer_udf("lemmatized")).drop(lemmatized_df["lemmatized"])


    ## Count Vectorizer
    cv = CountVectorizer(inputCol="stemmed", outputCol="vectors")
    cv_model = cv.fit(stemmed_df)
    cv_df = cv_model.transform(stemmed_df).drop(stemmed_df["stemmed"])
    cv_model.save("topic_modelling/cvmodel_pos")

    idf = IDF(inputCol="vectors",outputCol="tfidf")
    idf_model = idf.fit(cv_df)
    result = idf_model.transform(cv_df)

    result = result.select("review_id","business_id","tfidf")

    lda = LDA(featuresCol='tfidf', k=5, seed=42, maxIter=50)
    model = lda.fit(result)
    model.write().overwrite().save("topic_modelling/ldamodel_pos")
    transformed = model.transform(result)
    transformed.write.parquet("topic_modelling/review_topics_pos",mode="overwrite")
    spark.stop()

if __name__ == '__main__':
    spark = SparkSession.builder.appName('reddit average').getOrCreate()
    assert spark.version >= '2.3'
    stemmer = PorterStemmer()
    main()
