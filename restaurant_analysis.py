import sys
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

from wordcloud import WordCloud, STOPWORDS
import re
import string
import nltk
from textblob import TextBlob
from pyspark.ml.feature import *
from pyspark.sql import SparkSession, functions, types
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf, max
from pyspark.sql.functions import *

spark = SparkSession.builder.appName('example code').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as pl
import re, operator, string
from pyspark.ml.feature import StopWordsRemover
from textblob import Word
from wordcloud import ImageColorGenerator

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

# plot - Top customers satisfied restaurants on Yelp
df_yelp_review.registerTempTable("df_yelp_review")
top_restaurants = spark.sql("""SELECT name FROM df_yelp_review GROUP BY name ORDER BY COUNT(name) DESC LIMIT 20""")

top_restaurants_list = [(i.name) for i in top_restaurants.collect()]
df_review_top_rest = df_yelp_review.filter(df_yelp_review["name"].isin(top_restaurants_list))
df_review_top_rest = df_review_top_rest.groupby("name").agg(avg(df_review_top_rest.stars)).sort(desc("avg(stars)"))

pdf = df_review_top_rest.toPandas()
pdf = pdf.sort_values("avg(stars)",ascending=True)

pdf.plot("name",kind='barh',figsize=(15, 15))
plt.yticks(fontsize=18)
plt.title('Top customers satisfied restaurants on Yelp',fontsize=20)
plt.ylabel('Restaurants names', fontsize=18)
plt.xlabel('Reviews polarity', fontsize=18)
plt.savefig('visualization/top20_restaurant.png', format='png', dpi=1200)

# plot - No of useful, funny & cool reviews for restaurants

df_yelp_review.registerTempTable("df_yelp_review1")
df_review_UFC = spark.sql("""SELECT name,useful,funny,cool FROM df_yelp_review1""")
df_review_UFC = df_review_UFC.withColumn("useful", df_review_UFC["useful"].cast(IntegerType())).withColumn("funny", df_review_UFC["funny"].cast(IntegerType())).withColumn("cool", df_review_UFC["cool"].cast(IntegerType()))

df_review_UFC = df_review_UFC.groupby("name").mean("useful","funny","cool")
df_review_UFC = df_review_UFC.withColumnRenamed('avg(useful)','useful').withColumnRenamed('avg(funny)','funny').withColumnRenamed('avg(cool)','cool').sort(desc("useful"))

UFC = df_review_UFC.toPandas()
UFC = UFC[0:20]
UFC = UFC.sort_values("useful",ascending=True)

UFC.plot("name",kind='barh', figsize=(15, 14),width=0.7)
plt.yticks(fontsize=18)
plt.title('No of useful, funny & cool reviews for restaurants',fontsize=20)
plt.ylabel('Restaurants names', fontsize=18)
plt.yticks(fontsize=20)
plt.legend(fontsize=22)
plt.savefig('visualization/rev_votes_restaurant.png', format='png', dpi=1200)
