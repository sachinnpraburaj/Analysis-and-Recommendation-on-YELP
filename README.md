# Analysis-and-Recommendation-on-YELP
Analysis and Recommendation on YELP dataset

# Objective:
To provide useful insights using YELP dataset for businesses through big data analytics to determine strengths and weaknesses, so that existing owners and future business owners can make decision on new businesses or business expansion. Also to provide recommendation to both business owners and users by extensive analysis on data.

# Project Overview:
The project involves analysis on the dataset, visualization based on analysis and recommendations. Major modules of the project are:
1. Validation of reviews on businesses based on user information.
2. Classification of positive and negative reviews using Machine Learning techniques.
3. Recommending location based “buzzwords” to future business owners by analyzing positive reviews and negative reviews for a businesses in a state.
4. User-specific recommendations using user’s history of availed services. Recommendations are provided based on categories of the services, location of the business, user reviews and user ratings.

Analysis was done on the dataset to understand correlation between different metrics like - location of business and its success, etc. Analysis on business trends based on location, ratings, category and attributes of the business was performed. Trends of closed businesses was observed using user reviews and ratings.

Visualizations for the project were done using python libraries and are stored in [visualization](visualization) folder.

> Project presentation can be found at `Prezi WIN ARYD.exe` executed on a windows OS fr

# Steps for execution:
Dataset for the project should be downloaded from [Yelp dataset challenge](https://www.yelp.ca/dataset/download) and stored in yelp-dataset folder.
The codes should be executed in the following specified order in:
```
${SPARK_HOME}/bin/spark-submit business_etl.py
${SPARK_HOME}/bin/spark-submit user_etl.py
${SPARK_HOME}/bin/spark-submit review_classification.py
${SPARK_HOME}/bin/spark-submit review_etl.py
```
The following files can be executed in any order:
```
${SPARK_HOME}/bin/spark-submit user_recomm.py "'CxDOIDnH8gp9KXzpBHJYXw'"
# user name can be changed to obtain recommendations for different users
${SPARK_HOME}/bin/spark-submit user_analysis.py
${SPARK_HOME}/bin/spark-submit top_reviews.py
${SPARK_HOME}/bin/spark-submit business_analysis.py
${SPARK_HOME}/bin/spark-submit restaurant_analysis.py
${SPARK_HOME}/bin/spark-submit topic_mod_pos.py
${SPARK_HOME}/bin/spark-submit topic_mod_neg.py
${SPARK_HOME}/bin/spark-submit topics.py
${SPARK_HOME}/bin/spark-submit word_cloud.py
${SPARK_HOME}/bin/spark-submit ngram_word_cloud.py
```
Optional execution for converting data to json format for visualization:
```
${SPARK_HOME}/bin/spark-submit converttojson.py
```

# Files:

###### [business_etl.py](business_etl.py)
  -- business location - outliers removed using euclidean distance from avg location of businesses in state (Data Cleaning)

###### [user_etl.py](user_etl.py)
  -- users's location
  -- user validation score

###### [review_classification.py](review_classification.py)
  -- classification of reviews (Machine Learning)

###### [review_etl.py](review_etl.py)
  -- joined classes to reviews and dropped not so useful columns

###### [user_recomm.py](user_recomm.py)
  -- location based recommendations
  -- category based recommendations
  -- overall recommendations

###### [user_analysis.py](user_analysis.py)
  -- most availed category of business by an user
  -- average stars given by user for each category
  -- number of positive and negative reviews given by a user

###### [top_reviews.py](top_reviews.py)
  -- chose top 10 positive and top 10 negative reviews based on validation score for business with maximum reviews

###### [business_analysis.py](business_analysis.py)
  -- average review count and stars by city and category
  -- average review count and stars by state and category
  -- business attribute based analysis
  -- average stars for open and closed businesses
  -- top 15 business categories
  -- top 15 business categories - city-wise
  -- cities with most businesses
  -- businesses with more 5 star ratings

###### [restaurant_analysis.py](restaurant_analysis.py)
  -- top 20 restaurants on yelp (viz)
  -- restaurants with most funny, cool, useful reviews (viz)

###### [topic_mod_pos.py](topic_mod_pos.py)
  -- topic modeling using positive reviews for businesses in Pennsylvania

###### [topic_mod_neg.py](topic_mod_neg.py)
  -- topic modeling using negative reviews for businesses in Ontario

###### [topics.py](topics.py)
  -- extracted terms and topics from the model saved from topic modeling

###### [word_cloud.py](word_cloud.py)
  -- most frequent words from tips and review for Earl (viz)
  -- most frequent words from tips and review for Ontario (viz)
  -- most frequent words from tips and review for top 20 restaurants (viz)
  -- most frequent words from tips and review for bottom 20 restaurants (viz)

###### [ngram_word_cloud.py](ngram_word_cloud.py)
  -- wordcloud NGrams from tips review
  -- wordcloud NGrams from tips review for Arizona

###### [converttojson.py](converttojson.py)
  -- converting parquet ETLed files to JSON format for visualization purposes

# Folders:

###### [yelp-etl](yelp-etl)
  -- outputs after classification of reviews and etl steps on datasets will be stored

###### [visualization](visualization)
  -- outputs of all the visualizations will be stored here

###### [topic_modelling](topic_modelling)
  -- all results of topic modelling will be saved here

###### [analysis](analysis)
  -- all results of analysis will be stored here
