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
from pyspark.sql.functions import regexp_replace, col

spark = SparkSession.builder.appName('WordCloud').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as pl
import re, operator, string
from pyspark.ml.feature import StopWordsRemover
from textblob import Word
from wordcloud import ImageColorGenerator

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

#WordCloud for Specific restaurant Earl of Sandwich --> business based

df_yelp_tip.registerTempTable("df_yelp_tip")

Specific_restaurants_tip = spark.sql("""SELECT * FROM df_yelp_tip """)
Specific_restaurant = Specific_restaurants_tip.withColumn('name', regexp_replace('name', "\"\"\"", " "))

Specific_restaurant.registerTempTable("Specific_restaurant")
earl = spark.sql("""SELECT * FROM Specific_restaurant where  name == ' Earl of Sandwich ' """)
earl = earl.select("text")
tokenized_earl = Tokenizer(inputCol="text", outputCol="words")
tWords_earl = tokenized_earl.transform(earl)
SWremover_earl = StopWordsRemover(inputCol="words", outputCol="filtered")
SWremoved_earl = SWremover_earl.transform(tWords_earl).select("filtered")
FWords_earl = SWremoved_earl.rdd.flatMap(once)
WCount_earl = FWords_earl.reduceByKey(operator.add)
FreqWords_earl = WCount_earl.sortBy(lambda t: t[1], ascending = False).take(400)
FreqWordDict_earl = dict(FreqWords_earl)

mask_earl = np.array(Image.open("visualization/likesimba.png"))
wordcloud_earl = WordCloud(width =1600,height=800, background_color="white", max_words=1000, mask=mask_earl).generate_from_frequencies(FreqWordDict_earl)
image_colors_earl = ImageColorGenerator(mask_earl)

title_earl = 'Most frequent words from tips review for Earl of Sandwich'
plt.figure(figsize=[20,10],facecolor='k')
plt.imshow(wordcloud_earl.recolor(color_func=image_colors_earl),interpolation="bilinear")
plt.title(title_earl, size=25, y=1.01)
plt.axis("off")
plt.savefig("visualization/earl.png", format="png")

#WordCloud for restaurants in Ontario --> location based

df_yelp_tip.registerTempTable("df_yelp_tip")

ontario = spark.sql("""SELECT * FROM df_yelp_tip where  state == 'ON' """)
ontario = ontario.select("text")
tokenized_ontario = Tokenizer(inputCol="text", outputCol="words")
tWords_ontario = tokenized_earl.transform(ontario)
SWremover_ontario = StopWordsRemover(inputCol="words", outputCol="filtered")
SWremoved_ontario = SWremover_ontario.transform(tWords_ontario).select("filtered")
FWords_ontario = SWremoved_ontario.rdd.flatMap(once)
WCount_ontario = FWords_ontario.reduceByKey(operator.add)
FreqWords_ontario = WCount_ontario.sortBy(lambda t: t[1], ascending = False).take(400)
FreqWordDict_ontario = dict(FreqWords_ontario)

mask_ontario = np.array(Image.open("visualization/likesimba.png"))
wordcloud_ontario = WordCloud(width =1600,height=800, background_color="white", max_words=1000, mask=mask_ontario).generate_from_frequencies(FreqWordDict_earl)
image_colors_ontario = ImageColorGenerator(mask_ontario)

title_ontario = 'Most frequent words from tips review for Ontario'
plt.figure(figsize=[20,10],facecolor='k')
plt.imshow(wordcloud_ontario.recolor(color_func=image_colors_ontario),interpolation="bilinear")
plt.title(title_ontario, size=25, y=1.01)
plt.axis("off")
plt.savefig("visualization/ontario.png", format="png")

#WordCloud for top 20 restaurants --> top 20 based on business star rating
df_yelp_tip.registerTempTable("df_yelp_tip")
top_restaurants_tip = spark.sql("""SELECT name FROM df_yelp_tip GROUP BY name ORDER BY COUNT(name) DESC LIMIT 20""")

top_restaurants_list_tip = [(i.name) for i in top_restaurants_tip.collect()]
df_tip_top_rest = df_yelp_tip.filter(df_yelp_tip["name"].isin(top_restaurants_list_tip))
df_tip_top_rest = df_tip_top_rest.select("text")
tokenized = Tokenizer(inputCol="text", outputCol="words")
tWords = tokenized.transform(df_tip_top_rest)
SWremover = StopWordsRemover(inputCol="words", outputCol="filtered")
SWremoved = SWremover.transform(tWords).select("filtered")
FWords = SWremoved.rdd.flatMap(once)
WCount = FWords.reduceByKey(operator.add)
FreqWords = WCount.sortBy(lambda t: t[1], ascending = False).take(400)
FreqWordDict = dict(FreqWords)

mask = np.array(Image.open("visualization/like.jpg"))
wordcloud = WordCloud(width =1600,height=800, background_color="white", max_words=1000, mask=mask).generate_from_frequencies(FreqWordDict)
image_colors = ImageColorGenerator(mask)

title = 'Most frequent words from tips review for top restaurants'
plt.figure(figsize=[20,10],facecolor='k')
plt.imshow(wordcloud.recolor(color_func=image_colors),interpolation="bilinear")
plt.title(title, size=25, y=1.01)
plt.axis("off")
plt.savefig("visualization/top20.png", format="png")

#WordCloud for bottom 20 restaurants --> bottom 20 based on business star rating

df_yelp_tip.registerTempTable("df_yelp_tip")
bottom_restaurants_tip = spark.sql("""SELECT name FROM df_yelp_tip GROUP BY name ORDER BY COUNT(name) ASC LIMIT 20""")
bottom_restaurants_list_tip = [(i.name) for i in bottom_restaurants_tip.collect()]
df_tip_bottom_rest = df_yelp_tip.filter(df_yelp_tip["name"].isin(bottom_restaurants_list_tip))

stopwordList = ['a','able','about','across','after','all','almost','also','am','amazing','among','an','and','any','are','as','at','be','because','been','best','but','by','can',
'cannot','could','dear','did','do','does','either','else','ever','every','for','from','get','go','good,''got','great','had','has','have','he','her','hers','him','his','how','however','i','if','in',
'into','is','it','its','just','least','let','like','likely','may','me','might','most','must','my','neither','no','nor','not','of','off','often','on','only','or',
'other','our','own','rather','said','say','says','she','should','since','so','some','than','that','the','their','them','then','there','these','they','this','tis','to','too',
'twas','us','wants','was','we','were','what','when','where','which','while','who','whom','why','will','with','would','yet','you','your','yummy']
#new_stopwords = STOPWORDS.union(stopwordList)

df_tip_bottom_rest = df_tip_bottom_rest.select("text")
tokenizedbottom = Tokenizer(inputCol="text", outputCol="words")
tWordsbottom = tokenizedbottom.transform(df_tip_top_rest)
SWremoverbottom = StopWordsRemover(inputCol="words", outputCol="filtered", stopWords=stopwordList)
SWremovedbottom = SWremoverbottom.transform(tWords).select("filtered")
FWordsbottom = SWremovedbottom.rdd.flatMap(once)
WCountbottom = FWordsbottom.reduceByKey(operator.add)
FreqWordsbottom = WCountbottom.sortBy(lambda t: t[1], ascending = False).take(400)
FreqWordDictbottom = dict(FreqWordsbottom)

maskbottom = np.array(Image.open("visualization/dislike.jpg"))
wordcloudbottom = WordCloud(width =1600,height=800,background_color="white", max_words=1000, mask=maskbottom).generate_from_frequencies(FreqWordDictbottom)
image_colors_bottom = ImageColorGenerator(maskbottom)

titlebottom = 'Most frequent words from tips review for bottom 20 restaurants'
plt.figure(figsize=[20,10],facecolor='k')
plt.imshow(wordcloudbottom.recolor(color_func=image_colors_bottom))
plt.title(titlebottom, size=25, y=1.01)
plt.axis("off")
plt.savefig("visualization/bottom20.png", format="png")
