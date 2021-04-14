#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyspark
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

# create spark contexts

sc = pyspark.SparkContext()
sqlContext = SQLContext(sc)


# In[2]:


spark = SparkSession     .builder     .master("local[*]")    .appName("blogtext_preprocessing")     .getOrCreate()


# In[3]:


spark.version


# load blogtext.csv

# In[4]:


blogtext_df = spark.read.csv('/spring2021/project1/blogtext.csv',header=True, inferSchema='true')

blogtext_df.show(5)


# In[5]:


blogtext_df.printSchema()


# In[6]:


type(blogtext_df)


# In[7]:


blogtext_df.count()


# setup pyspark udf functions

# In[8]:


from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import nltk
import preproc as pp

# preproc is a python file that has functions defined to do the "text" column preprocessing.

# Register all the functions in Preproc with Spark Context

remove_stops_udf = udf(pp.remove_stops, StringType())
remove_features_udf = udf(pp.remove_features, StringType())
tag_and_remove_udf = udf(pp.tag_and_remove, StringType())
lemmatize_udf = udf(pp.lemmatize, StringType())


# packages downloaded from nltk

# In[9]:


#import nltk
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')


# remove stop words

# In[10]:


# get the raw columns
raw_cols = blogtext_df.columns

rm_stops_df = blogtext_df.select(raw_cols)                   .withColumn("stop_text", remove_stops_udf(blogtext_df["text"]))


# remove column "text" from "rm_stops_df".now we are going to use "stop_text" column which is the one with all the stopwords removed.

# In[11]:


from functools import reduce
from pyspark.sql import DataFrame

rm_stops_df = reduce(DataFrame.drop, ['text'], rm_stops_df)
raw_cols = rm_stops_df.columns


# remove irrelevant features

# In[12]:


rm_features_df = rm_stops_df.select(raw_cols)                            .withColumn("feat_text",                             remove_features_udf(rm_stops_df["stop_text"]))


# In[13]:


#remove column "stop_text" from "rm_stops_df".
#now we are going to use "feat_text" column which is the one with all the unwanted features removed.

rm_features_df = reduce(DataFrame.drop, ['stop_text'], rm_features_df)

raw_cols = rm_features_df.columns


# tag the words

# In[14]:


tagged_df = rm_features_df.select(raw_cols)                           .withColumn("tagged_text",                            tag_and_remove_udf(rm_features_df.feat_text))


# In[15]:


#remove column "feat_text" from "rm_stops_df".
#now we are going to use "tagged_text" column which is the one with all the unwanted features removed.

tagged_df = reduce(DataFrame.drop, ['feat_text'], tagged_df)

raw_cols = tagged_df.columns


# lemmatization of words

# In[16]:


lemm_df = tagged_df.select(raw_cols)                    .withColumn("text", lemmatize_udf(tagged_df["tagged_text"]))


# In[17]:


#remove column "tagged_text" from "rm_stops_df".
#now we are going to use "text" column which is the one with all the unwanted features removed.

lemm_df = reduce(DataFrame.drop, ['tagged_text'], lemm_df)

raw_cols = lemm_df.columns


# In[18]:


# lemm_df is now clean

lemm_df.show(5)


# Group data by topic

# In[19]:


from pyspark.sql.functions import col

topic_count = lemm_df.groupBy("topic").count().orderBy(col("count").desc())

topic_count.count()


# There are 40 interesting topics in the blog.

# In[20]:


topic_count.show(truncate=False)


# plot to show all the topics in the blog

# In[22]:


import matplotlib.pyplot as plt
import pandas as pd
df_pandas = topic_count.toPandas()
df_pandas.plot(figsize = (10,5), kind='bar', x='topic', y='count', colormap='winter_r')
plt.show()


# indUnk and Student are the topics highly discussed. Other topics have less than 50,000 records.

# filter and analyse topic "indUnk"

# In[23]:


topic_indUnk = lemm_df.select('text').filter(lemm_df.topic == "indUnk") 
topic_indUnk.count()


# converting the filtered dataframe to rdd

# In[24]:


topic_indUnk_rdd = topic_indUnk.rdd.map(list)
topic_indUnk_rdd.take(2)


# removing the square brackets at the begin and end of each row

# In[25]:


import re

topic_indUnk_rdd = topic_indUnk_rdd.map(lambda x: re.sub('\[|\]', '', str(x)))
topic_indUnk_rdd.take(2)


# split rdd

# In[26]:


split_rdd = topic_indUnk_rdd.flatMap(lambda line: line.split(" "))
split_rdd.take(20)


# In[ ]:


#print("Total number of words in 'indUnk' topic:", split_rdd.count())


# map each word with 1, then reduce by key(words), and finally sort by key(count)

# In[31]:


map_rdd = split_rdd.map(lambda x: (x,1))
reduce_rdd = map_rdd.reduceByKey(lambda a,b: a+b)
sort_rdd = reduce_rdd.map(lambda x: (x[1],x[0])).sortByKey()


# show word count for topic "indUnk"

# In[33]:


#for word in sort_rdd.take(20):
    #print(word[1], word[0])
sort_rdd.take(20)    


# In[68]:


sort_df = spark.createDataFrame(sort_rdd)


# In[69]:


sort_df.show()


# In[70]:


sort_df.select("_2").show()


# In[34]:


#import matplotlib.pyplot as plt

#num_bins = 50
#n, bins, patches = plt.hist(sort_rdd.collect(), num_bins, normed=1, facecolor='green', alpha=0.5)


# filter and analyse topic "Student".following the same steps as we did with topic "indUnk"

# In[35]:


topic_student = lemm_df.select('text').filter(lemm_df.topic == "Student") 
topic_student.count()


# In[36]:


topic_student_rdd = topic_student.rdd.map(list)
topic_student_rdd.take(2)


# In[37]:


#import re

topic_student_rdd = topic_student_rdd.map(lambda x: re.sub('\[|\]', '', str(x)))
topic_student_rdd.take(2)


# In[38]:


split_rdd1 = topic_student_rdd.flatMap(lambda line: line.split(" "))
split_rdd1.take(20)


# In[39]:


print("Total number of words in 'Student' topic:", split_rdd1.count())


# In[40]:


map_rdd1 = split_rdd1.map(lambda x: (x,1))
reduce_rdd1 = map_rdd1.reduceByKey(lambda a,b: a+b)
sort_rdd1 = reduce_rdd1.map(lambda x: (x[1],x[0])).sortByKey()


# show word count for topic "Student"

# In[41]:


#for word in sort_rdd1.take(20):
    #print(word[1], word[0])
sort_rdd1.take(20)    


# analyse topic "Technology"

# In[42]:


topic_tech = lemm_df.select('text').filter(lemm_df.topic == "Technology") 
topic_tech.count()


# In[43]:


topic_tech_rdd = topic_tech.rdd.map(list)
topic_tech_rdd.take(2)


# In[44]:


topic_tech_rdd = topic_tech_rdd.map(lambda x: re.sub('\[|\]', '', str(x)))
topic_tech_rdd.take(2)


# In[45]:


split_rdd2 = topic_tech_rdd.flatMap(lambda line: line.split(" "))
split_rdd2.take(20)


# In[46]:


print("Total number of words in 'Technology' topic:", split_rdd2.count())


# In[48]:


map_rdd2 = split_rdd2.map(lambda x: (x,1))
reduce_rdd2 = map_rdd2.reduceByKey(lambda a,b: a+b)
sort_rdd2 = reduce_rdd2.map(lambda x: (x[1],x[0])).sortByKey()


# show word count for topic "Technology"

# In[49]:


#for word in sort_rdd2.take(50):
    #print(word[1], word[0])
sort_rdd2.take(20)    


# In[71]:


sort_df2 = spark.createDataFrame(sort_rdd2)
sort_df2.select("_2").show()


# analyse topic "Arts"

# In[50]:


topic_arts = lemm_df.select('text').filter(lemm_df.topic == "Arts") 
topic_arts.count()


# In[51]:


topic_arts_rdd = topic_arts.rdd.map(list)
topic_arts_rdd.take(2)


# In[52]:


topic_arts_rdd = topic_arts_rdd.map(lambda x: re.sub('\[|\]', '', str(x)))
topic_arts_rdd.take(2)


# In[53]:


split_rdd3 = topic_arts_rdd.flatMap(lambda line: line.split(" "))
split_rdd3.take(20)


# In[54]:


#print("Total number of words in 'Arts' topic:", split_rdd3.count())


# In[55]:


map_rdd3 = split_rdd3.map(lambda x: (x,1))
reduce_rdd3 = map_rdd3.reduceByKey(lambda a,b: a+b)
sort_rdd3 = reduce_rdd3.map(lambda x: (x[1],x[0])).sortByKey()


# show word count for topic "Arts"

# In[56]:


#for word in sort_rdd3.take(50):
    #print(word[1], word[0])
sort_rdd3.take(20)    


# In[72]:


sort_df3 = spark.createDataFrame(sort_rdd3)
sort_df3.select("_2").show()


# analyse topic "Education"

# In[57]:


topic_edu = lemm_df.select('text').filter(lemm_df.topic == "Education") 
topic_edu.count()


# In[58]:


topic_edu_rdd = topic_edu.rdd.map(list)
topic_edu_rdd.take(2)


# In[59]:


topic_edu_rdd = topic_edu_rdd.map(lambda x: re.sub('\[|\]', '', str(x)))
topic_edu_rdd.take(2)


# In[60]:


split_rdd4 = topic_edu_rdd.flatMap(lambda line: line.split(" "))
split_rdd4.take(20)


# In[61]:


#print("Total number of words in 'Education' topic:", split_rdd4.count())


# In[62]:


map_rdd4 = split_rdd4.map(lambda x: (x,1))
reduce_rdd4 = map_rdd4.reduceByKey(lambda a,b: a+b)
sort_rdd4 = reduce_rdd4.map(lambda x: (x[1],x[0])).sortByKey()


# show word count for topic "Education"

# In[63]:


#for word in sort_rdd4.take(50):
    #print(word[1], word[0])
sort_rdd4.take(20)    


# In[65]:


sort_df4 = spark.createDataFrame(sort_rdd4) 


# In[66]:


sort_df4.show()


# In[67]:


sort_df4.select("_2").show()


# In[ ]:




