from os import path, getcwd
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import plotly
import plotly.tools as tls
import plotly.graph_objs as go
from plotly.graph_objs import *
import plotly.tools as tls
import plotly.figure_factory as fig_fact
plotly.tools.set_config_file(world_readable=True, sharing='public')
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import warnings
warnings.filterwarnings('ignore')

from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as pl
import re, operator, string
from pyspark.ml.feature import StopWordsRemover
from textblob import Word
from wordcloud import ImageColorGenerator

import string
import re, operator
from pyspark.sql import SparkSession, functions, types
from textblob import Word
from wordcloud import ImageColorGenerator
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import *
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
import pyspark.ml.feature as ft
spark = SparkSession.builder.appName("yelp Ngram").getOrCreate()

def once(line):
    for x in line[0]:
        Wsep = re.compile(r'[%s]+' % re.escape(string.punctuation))
        s = Wsep.split(x)
        for w in s:
            if (len(w)>1):
                yield (Word(w).lemmatize(), 1)


df_yelp_tip = spark.read.json('yelp-dataset/yelp_academic_dataset_tip.json')
df_yelp_business = spark.read.json('yelp-dataset/yelp_academic_dataset_business.json')
df_yelp_review = spark.read.json('yelp-dataset/yelp_academic_dataset_review.json')

df_category_split = df_yelp_business.select('categories','business_id')
df_category_split = df_category_split.withColumnRenamed('business_id','business_id_1')
df_category_split = df_category_split.withColumnRenamed('categories','categories_split')
df_category_split = df_category_split.withColumn('categories1',functions.split(df_category_split['categories_split'], ';').getItem(0)).withColumn('categories2',functions.split(df_category_split['categories_split'], ';').getItem(1)).withColumn('categories3',functions.split(df_category_split['categories_split'], ';').getItem(2))

df_yelp_business = df_yelp_business.join(df_category_split, df_yelp_business['business_id'] == df_category_split['business_id_1'])
df_yelp_business = df_yelp_business.drop("business_id_1")
df_yelp_business = df_yelp_business.drop("categories_split")

df_yelp_business_restaurants = df_yelp_business.filter((df_yelp_business['categories1'] == 'Restaurants') |(df_yelp_business['categories2'] == 'Restaurants') | (df_yelp_business['categories3'] == 'Restaurants'))
df_yelp_business_restaurants = df_yelp_business_restaurants.withColumnRenamed('stars', 'stars_bus')
df_yelp_business_restaurants = df_yelp_business_restaurants.withColumnRenamed('business_id','business_id_rest')

df_yelp_review = df_yelp_review.join(df_yelp_business_restaurants, df_yelp_review['business_id'] == df_yelp_business_restaurants['business_id_rest'])

df_yelp_tip = df_yelp_tip.join(df_yelp_business_restaurants, df_yelp_tip['business_id'] == df_yelp_business_restaurants['business_id_rest'])

df_yelp_review.registerTempTable("df_yelp_review")
top_restaurants = spark.sql("""SELECT name FROM df_yelp_review GROUP BY name ORDER BY COUNT(name) DESC LIMIT 20""")

top_restaurants_list = [(i.name) for i in top_restaurants.collect()]
df_review_top_rest = df_yelp_review.filter(df_yelp_review["name"].isin(top_restaurants_list))

df_review_top_rest = df_review_top_rest.select("text").limit(10000)

tokenizer = ft.RegexTokenizer(
    inputCol='text',
    outputCol='word',
    pattern='\s+|[,.\"]')

tok = tokenizer \
    .transform(df_review_top_rest) \
    .select('word')

stopwords = ft.StopWordsRemover(
    inputCol=tokenizer.getOutputCol(),
    outputCol='input_stop')

ngram = ft.NGram(n=2,
    inputCol=stopwords.getOutputCol(),
    outputCol="nGrams")

pipeline = Pipeline(stages=[tokenizer, stopwords, ngram])

data_ngram = pipeline \
    .fit(df_review_top_rest) \
    .transform(df_review_top_rest)

data_ngram = data_ngram.select('nGrams')

FWords = data_ngram.rdd.flatMap(once)
WCount = FWords.reduceByKey(operator.add)
FreqWords = WCount.sortBy(lambda t: t[1], ascending = False).take(400)
FreqWordDict = dict(FreqWords)

#print(FreqWordDict)

mask = np.array(Image.open("visualization/likesimba.png"))
wordcloud = WordCloud(width =1600,height=800, background_color="white", max_words=1000, mask=mask).generate_from_frequencies(FreqWordDict)
image_colors = ImageColorGenerator(mask)

title = 'WC NGrams from tips review'
plt.figure(figsize=[20,10],facecolor='k')
plt.imshow(wordcloud.recolor(color_func=image_colors),interpolation="bilinear")
plt.title(title, size=25, y=1.01)
plt.axis("off")
plt.savefig("visualization/ngramtop.png", format="png")


df_yelp_tip.registerTempTable("df_yelp_tip")

Arizona = spark.sql("""SELECT * FROM df_yelp_tip where  state == 'AZ' """)
Arizona = Arizona.select("text")

tokenizer = ft.RegexTokenizer(
    inputCol='text',
    outputCol='word',
    pattern='\s+|[,.\"]')

tok = tokenizer \
    .transform(Arizona) \
    .select('word')


stopwords = ft.StopWordsRemover(
    inputCol=tokenizer.getOutputCol(),
    outputCol='input_stop')

ngram = ft.NGram(n=2,
    inputCol=stopwords.getOutputCol(),
    outputCol="nGrams")

pipeline = Pipeline(stages=[tokenizer, stopwords, ngram])

data_ngram = pipeline \
    .fit(Arizona) \
    .transform(Arizona)

data_ngram = data_ngram.select('nGrams')

FWords = data_ngram.rdd.flatMap(once)
WCount = FWords.reduceByKey(operator.add)
FreqWords = WCount.sortBy(lambda t: t[1], ascending = False).take(400)
FreqWordDict = dict(FreqWords)

#print(FreqWordDict)

mask = np.array(Image.open("visualization/likesimba.png"))
wordcloud = WordCloud(width =1600,height=800, background_color="white", max_words=1000, mask=mask).generate_from_frequencies(FreqWordDict)
image_colors = ImageColorGenerator(mask)

title = 'WC NGrams from tips review for Arizona'
plt.figure(figsize=[20,10],facecolor='k')
plt.imshow(wordcloud.recolor(color_func=image_colors),interpolation="bilinear")
plt.title(title, size=25, y=1.01)
plt.axis("off")
plt.savefig("visualization/Arizona.png", format="png")
