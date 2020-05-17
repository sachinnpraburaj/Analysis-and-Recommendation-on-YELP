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

spark = SparkSession.builder.appName('reddit average').getOrCreate()
assert spark.version >= '2.3'

review_etl = spark.read.parquet("yelp-etl/review_etl")
review_etl.coalesce(1).write.json("forViz/review_etl",mode="overwrite")
business_etl = spark.read.parquet("yelp-etl/business_etl")
business_etl.coalesce(1).write.json("forViz/business_etl",mode="overwrite")
user_etl = spark.read.parquet("yelp-etl/user_etl")
user_etl.coalesce(1).write.json("forViz/user_etl",mode="overwrite")
