# Databricks notebook source
# MAGIC %md # wk4 Demo - Intro to Spark
# MAGIC __`MIDS w261: Machine Learning at Scale | UC Berkeley School of Information | Spring 2019`__
# MAGIC 
# MAGIC Last week we saw a number of design patterns in Hadoop MapReduce. This week we will look at the limitations of Hadoop MapReduce when it comes to running iterative jobs and preview the advantages of modern distributed compuation frameworks like Spark. By abstracting away many of the parallelization details Spark provides a flexible interface for the programmer. However a word of warning: don't let the ease of implementation lull you into complacency, scalable solutions still require attention to the details of smart algorithm design. 
# MAGIC 
# MAGIC In class today we'll get some practice working with Spark RDDS. We'll use Spark to re-implement each of the tasks that you performed using the Command Line or Hadoop Streaming in weeks 1-3 of the course. Our goal is to get you up to speed and coding in Spark as quickly as possible; this is by no means a comprehensive tutorial. By the end of today's demo you should be able to:  
# MAGIC * ... __initialize__ a `SparkSession` in a local NB and use it to run a Spark Job.
# MAGIC * ... __access__ the Spark Job Tracker UI.
# MAGIC * ... __describe__ and __create__ RDDs from files or local Python objects.
# MAGIC * ... __explain__ the difference between actions and transformations.
# MAGIC * ... __decide__ when to `cache` or `broadcast` part of your data.
# MAGIC * ... __implement__ Word Counting, Sorting and Naive Bayes in Spark. 
# MAGIC 
# MAGIC __`NOTE:`__ Although RDD successor datatype, Spark dataframes, are becoming more common in production settings we've made a deliberate choice to teach you RDDs first beause building homegrown algorithm implementations is crucial to developing a deep understanding of machine learning and parallelization concepts -- which is the goal of this course. We'll still touch on dataframes in Week 5 when talking about Spark efficiency considerations and we'll do a deep dive into Spark dataframes and streaming solutions in Week 12.
# MAGIC 
# MAGIC __`Additional Resources:`__ The offical documentation pages offer a user friendly overview of the material covered in this week's readings: [Spark RDD Programming Guide](https://spark.apache.org/docs/latest/rdd-programming-guide.html#rdd-programming-guide).
# MAGIC 
# MAGIC __RDD API docs__: https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD

# COMMAND ----------

# MAGIC %md ### Notebook Set-Up

# COMMAND ----------

# imports
import re
import numpy as np
import pandas as pd

# COMMAND ----------

# MAGIC %md ### Run the next three cells to create your DEMO4 directory 

# COMMAND ----------

# MAGIC %scala
# MAGIC //*******************************************
# MAGIC // GET USERNAME AND USERHOME
# MAGIC //*******************************************
# MAGIC val tags = com.databricks.logging.AttributionContext.current.tags
# MAGIC 
# MAGIC // Get the user's name
# MAGIC val name = tags.getOrElse(com.databricks.logging.BaseTagDefinitions.TAG_USER, java.util.UUID.randomUUID.toString.replace("-", ""))
# MAGIC val username = if (name != "unknown") name else dbutils.widgets.get("databricksUsername")
# MAGIC 
# MAGIC val userhome = s"dbfs:/user/$username"
# MAGIC 
# MAGIC // println(userhome)
# MAGIC 
# MAGIC val userDataFrame = Seq((name,username,userhome)).toDF("name","username","userhome")
# MAGIC userDataFrame.createOrReplaceTempView( "userTable" )

# COMMAND ----------

# From here, you can specify the path like 
result = sqlContext.sql('select userhome from userTable')
userhome = result.first()[0]
demo4_path = userhome + "/demo4/" 
dbutils.fs.mkdirs(demo4_path)

# COMMAND ----------

dbutils.fs.put(demo4_path+'test.txt',"hello world",True)
display(dbutils.fs.ls(demo4_path))

# COMMAND ----------

# MAGIC %md ### Load the data
# MAGIC Today we'll mostly be working with toy examples & data created on the fly in Python. However at the end of this demo we'll revisit Word Count & Naive Bayes using some of the data from weeks 1-3. The _Alice in Wonderland_ text file has been pre-loaded into DBFS. Run the following cells to inspect this file and store its location in a python variable ALICE_TXT.

# COMMAND ----------

# MAGIC %md #### Using the 'head' shell command, we can take a peek at the top 10 lines of our dataset to confirm it is what we think it is.

# COMMAND ----------

# MAGIC %sh head /dbfs/MIDS-W261/HW2/11-0.txt

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Here we'll store the location of our dataset in a python variable
# MAGIC By default, Spark understands the / as /dbfs, e.g. /MIDS-W261 == dbfs://MIDS-W261

# COMMAND ----------

ALICE_TXT = "/MIDS-W261/HW2/11-0.txt"

# COMMAND ----------

dbutils.fs.head(ALICE_TXT)

# COMMAND ----------

# MAGIC %md ### Now let's create a couple of toy examples on the fly
# MAGIC And store them in dbfs

# COMMAND ----------

dbutils.fs.put(demo4_path+"chineseTrain.txt", 
"""D1	1		Chinese Beijing Chinese
D2	1		Chinese Chinese Shanghai
D3	1		Chinese Macao
D4	0		Tokyo Japan Chinese""", True)

# COMMAND ----------

dbutils.fs.put(demo4_path+"chineseTest.txt", 
"""D5	1		Chinese Chinese Chinese Tokyo Japan
D6	1		Beijing Shanghai Trade
D7	0		Japan Macao Tokyo
D8	0		Tokyo Japan Trade""", True)


# COMMAND ----------

# naive bayes toy example data paths
TRAIN_PATH = demo4_path+"chineseTrain.txt"
TEST_PATH = demo4_path+"chineseTest.txt"

# COMMAND ----------

# MAGIC %md # Exercise 1. Getting started with Spark. 
# MAGIC For week 4 you read Ch 3-4 from _Learning Spark: Lightning-Fast Big Data Analysis_ by Karau et. al. as well as a few blog posts that set the stage for Spark. From these readings you should be familiar with each of the following terms:
# MAGIC 
# MAGIC * __Spark session__
# MAGIC * __Spark context__
# MAGIC * __driver program__
# MAGIC * __executor nodes__
# MAGIC * __resilient distributed datasets (RDDs)__
# MAGIC * __pair RDDs__
# MAGIC * __actions__ and __transformations__
# MAGIC * __lazy evaluation__
# MAGIC 
# MAGIC The first code block below shows you that a `SparkSession` is already running in this cluster. Additionally, we'll save the sparkContext in a variable `sc` so we can use the RDD API. Next we show a simple example of creating and transforming a Spark RDD. Let's use this as a quick vocab review before we dive into more interesting examples. 

# COMMAND ----------

spark

# COMMAND ----------

sc = spark.sparkContext

# COMMAND ----------

# sc.getConf().getAll()

# COMMAND ----------

# a small example
myData_RDD = sc.parallelize(range(1,100))
squares_RDD = myData_RDD.map(lambda x: (x,x**2))
oddSquares = squares_RDD.filter(lambda x: x[1] % 2 == 1)

# COMMAND ----------

oddSquares.take(5)

# COMMAND ----------

# MAGIC %md  > __DISCUSSION QUESTIONS:__ For each key term from the reading, briefly explain what it means in the context of this demo code. Specifically:
# MAGIC  * _What is the 'driver program' here?_
# MAGIC  * _What does the spark context do? Do we have 'executors' per se?_
# MAGIC  * _List all RDDs and pair RDDs present in this example._
# MAGIC  * _List all transformations present in this example._
# MAGIC  * _List all actions present in this example._
# MAGIC  * _What does the concept of 'lazy evaluation' mean about the time it would take to run each cell in the example?_
# MAGIC  * _If we were working on a cluster, where would each transformation happen? would the data get shuffled?_

# COMMAND ----------

displayHTML("<img src='files/DEMO4/spark-context.png' style='width:400px;'/>")

# COMMAND ----------

# MAGIC %md # Exercise 2. RDD transformations warm ups.
# MAGIC 
# MAGIC Here are some more examples of Spark transformations and actions. For each task below, we've provided a few different implementations. Read each example and discuss the differences. Is one implementation better than the other or are the differences cosmetic? You may wish to discuss:
# MAGIC * the format of the data after each transformation
# MAGIC * memory usage (on executor nodes & in the driver)
# MAGIC * time complexity
# MAGIC * amount of network transfer
# MAGIC * whether or not the data will get shuffled
# MAGIC * coding efficiency & readability  
# MAGIC 
# MAGIC Although we're working with tiny demo examples for now, try to imagine how the same code would operate if we were running a large job on a cluster. To aid in your analysis, navigate to the Spark UI (available at http://localhost:4040). To start, you should see a single job -- the job from Exercise 1. Click on the job description to view the DAG for that job. Check back with this UI as you run each version of the tasks below (__Note__: _the stages tab may be particularly helpful when making your comparisons_).
# MAGIC 
# MAGIC #### a) Multiples of 5 and 7

# COMMAND ----------

# VERSION 1
dataRDD = sc.parallelize(range(1,100))
fivesRDD = dataRDD.filter(lambda x: x % 5 == 0)
sevensRDD = dataRDD.filter(lambda x: x % 7 == 0)
result = fivesRDD.intersection(sevensRDD)
result.collect()

# COMMAND ----------

# VERSION 2
dataRDD = sc.parallelize(range(1,100))
result = dataRDD.filter(lambda x: x % 5 == 0)\
                .filter(lambda x: x % 7 == 0)
result.collect()

# COMMAND ----------

# VERSION 3
dataRDD = sc.parallelize(range(1,100))
result = dataRDD.filter(lambda x: x % 7 == 0 and x % 5 == 0)
result.collect()

# COMMAND ----------

# MAGIC %md >__DISCUSSION QUESTION:__ 
# MAGIC * What is the task here? Compare/contrast these three implementations.  
# MAGIC * Which of these versions require a shuffle? How do you know?

# COMMAND ----------

# MAGIC %md #### b) Pig Latin Translator

# COMMAND ----------

poem = ["A bear however hard he tries", 
        "Grows tubby without exercise", 
        "said AA Milne"]

# COMMAND ----------

# VERSION 1
def translate(sent):
    words = [w[1:] + w[0] + '-ay' for w in sent.lower().split()]
    return ' '.join(words)

poemRDD = sc.parallelize(poem)
result = poemRDD.map(translate)\
                .reduce(lambda x,y: x + ' ' + y)
print(result)

# COMMAND ----------

# VERSION 2
def translate(wrd):
    return wrd[1:] + wrd[0] + '-ay'

poemRDD = sc.parallelize(poem)
result = poemRDD.flatMap(lambda x: x.lower().split())\
                .map(translate)\
                .reduce(lambda x,y: x + ' ' + y)
print(result)

# COMMAND ----------

# MAGIC %md >__DISCUSSION QUESTION:__ What is the task here? Compare/contrast these two implementations.

# COMMAND ----------

# MAGIC %md #### c) Average Monthly Purchases

# COMMAND ----------

shoppingList = ["JAN: 5 apples, 15 oranges",
                "FEB: 10 apples, 10 oranges",
                "MAR: 3 apples, 1 oranges",
                "APR: 6 apples, 2 oranges"]

# COMMAND ----------

# helper function
def parseShopping(line):
    """Parse each month's shopping list string into a key-value iterator."""
    month, items = line.split(':')
    items = [item.strip().split(' ') for item in items.split(',')]
    return [(i[1], int(i[0])) for i in items]

# COMMAND ----------

# VERSION 1  (example 4-7 from Learning Spark)
shoppingRDD = sc.parallelize(shoppingList)
result = shoppingRDD.flatMap(lambda x: parseShopping(x))\
                    .mapValues(lambda x: (x,1))\
                    .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))\
                    .mapValues(lambda x: x[0]/float(x[1]))
result.collect()

# COMMAND ----------

# VERSION 2 (example 4-12 from Learning Spark)
shoppingRDD = sc.parallelize(shoppingList)
result = shoppingRDD.flatMap(lambda x: parseShopping(x))\
                    .combineByKey(lambda x: (x,1),       # action for new key
                                  lambda x, y: (x[0] + y, x[1] + 1), # action for repeat key
                                  lambda x, y: (x[0] + y[0], x[1] + y[1]))\
                    .mapValues(lambda x: x[0]/float(x[1]))
result.collect()

# COMMAND ----------

# VERSION 3
shoppingRDD = sc.parallelize(shoppingList)
result = shoppingRDD.flatMap(lambda x: parseShopping(x))\
                    .groupByKey()\
                    .mapValues(lambda x: sum(x)/float(len(x)))
result.collect()

# COMMAND ----------

# MAGIC %md >__DISCUSSION QUESTION:__ What is the task here? Compare/contrast these three implementations.

# COMMAND ----------

# MAGIC %md #  Exercise 3. cache()-ing
# MAGIC 
# MAGIC In exercise 2 you saw how Spark builds an execution plan (DAG) so that transformations are evaluated lazily when triggerd by an action. In more complex DAGs you may need to reuse the contents of an RDD for multiple downstream operations. In such cases we'd like to avoid duplicating the computation of that intermediate result. Spark offers a few different options to persist an RDD in memory on the executor node where it is stored. Of these the most common is `cache()` (you'll read about others next week in ch 5 from _High Performance Spark_). Lets briefly look at how to `cache()` an RDD and discus when doing so is to your advantage.

# COMMAND ----------

# initialize data
dataRDD = sc.parallelize(np.random.random_sample(1000))   

# COMMAND ----------

# perform some transformations
data2X= dataRDD.map(lambda x: x*2)
dataGreaterThan1 = data2X.filter(lambda x: x > 1.0)
cachedRDD = dataGreaterThan1.cache()

# COMMAND ----------

# count results less than 1
cachedRDD.filter(lambda x: x<1).count()

# COMMAND ----------

# count results greater than 1
cachedRDD.filter(lambda x: x>1).count()

# COMMAND ----------

# look at 10 results
for line in cachedRDD.take(10):
    print(line)

# COMMAND ----------

# look at top 10 results
for line in cachedRDD.top(10):
    print(line)

# COMMAND ----------

# look at top 10 results
for line in cachedRDD.takeOrdered(10):
    print(line)

# COMMAND ----------

# MAGIC %md > __DISCUSSION QUESTIONS:__
# MAGIC * How many total actions are there in the 7 cells above?
# MAGIC * If we hadn't cached the `dataGreaterThan1` RDD what would happen each time we call an action?
# MAGIC * How does `cache()` change what the framework does? 
# MAGIC * When does it _not_ make sense to `cache()` an intermediate result?

# COMMAND ----------

# MAGIC %md # Exercise 4. broadcast()-ing
# MAGIC 
# MAGIC Another challenge we faced when designing Hadoop MapReduce jobs was the challenge of making key pieces of information available to multiple nodes so that certain computations can happen in parallel. In Hadoop Streaming we resolved this challenge using custom partition keys and the order inversion pattern. In Spark we'll use broadcast variables -- read only objects that Spark will ship to all nodes where they're needed. Here's a brief example of how to create and access a broadcast variable.

# COMMAND ----------

# MAGIC %md Run the following cell to create our sample data files: a list of customers & a list of cities.

# COMMAND ----------

#Write a csv file in dbfs. Parameters (dbfs filename, contents, overwrite)
dbutils.fs.put(demo4_path+"customers.csv", 
"""Quinn Frank,94703
Morris Hardy,19875
Tara Smith,12204
Seth Mitchell,38655
Finley Cowell,10005
Cory Townsend,94703
Mira Vine,94016
Lea Green,70118
V Neeman,16604
Tvei Qin,70118""", True)

# COMMAND ----------

display(dbutils.fs.ls(demo4_path))

# COMMAND ----------

dbutils.fs.head('dbfs:/user/kylehamilton@ischool.berkeley.edu/hw3/customers.csv')

# COMMAND ----------

# MAGIC %sh head /dbfs/user/kylehamilton@ischool.berkeley.edu/hw3/customers.csv

# COMMAND ----------

zipCodesTxt = """94703,Berkeley,CA
94016,San Francisco,CA
10005,New York,NY
12204,Albany,NY
38655,Oxford,MS
70118,New Orleans,LA"""

# COMMAND ----------

zipCodesTxt

# COMMAND ----------

# MAGIC %md Spark Job to count customers by state.

# COMMAND ----------

# load customers from file
dataRDD = sc.textFile(demo4_path+"customers.csv")

# COMMAND ----------

# create a look up dictionary to map zip codes to state abbreviations
zipCodes = {l.split(',')[0]:l.split(',')[2] for l in zipCodesTxt.split('\n')}
print(zipCodes)


# COMMAND ----------

# Broadcast the zipcodes
zipCodes = sc.broadcast(zipCodes) 

# COMMAND ----------

# count by state
result = dataRDD.map(lambda x: x.split(',')[1])\
                .map(lambda x: (zipCodes.value.get(x,'n/a'),1))\
                .reduceByKey(lambda a, b: a + b)

# COMMAND ----------

# take a look
result.collect()

# COMMAND ----------

# MAGIC %md > __DISCUSSION QUESTIONS:__
# MAGIC * What does broadcasting achieve here?
# MAGIC * Why not just encapsulate our variables in a function closure instead?
# MAGIC * When would it be a bad idea to broadcast a supplemental table like our list of zip codes?
# MAGIC * Note that we are working in local mode through out this notebook. What happens if you comment out the line where we broadcast the zip code dictionary? What would happen if you were working on a cluster?

# COMMAND ----------

# MAGIC %md # Exercise 5. Accumulators
# MAGIC 
# MAGIC Accumulators are Spark's equivalent of Hadoop counters. Like broadcast variables they represent shared information across the nodes in your cluster, but unlike broadcast variables accumulators are _write-only_ ... in other words you can only access their values in the driver program and not on your executors (where transformations are applied). As convenient as this sounds, there are a few common pitfalls to avoid. Let's take a look.
# MAGIC 
# MAGIC Run the following cell to create a sample data file representing a list of `studentID, courseID, final_grade`...

# COMMAND ----------

dbutils.fs.put(demo4_path+"grades.csv", 
"""10001,101,98
10001,102,87
10002,101,75
10002,102,55
10002,103,80
10003,102,45
10003,103,75
10004,101,90
10005,101,85
10005,103,60""", True)


# COMMAND ----------

# MAGIC %md Suppose we want to compute the average grade by course and student while also tracking the number of failing grades awarded. We might try something like this:

# COMMAND ----------

# initialize an accumulator to track failing grades
nFailing = sc.accumulator(0)

# COMMAND ----------

# function to increment the accumulator as we read in the data
def parse_grades(line, accumulator):
    """Helper function to parse input & track failing grades."""
    student,course,grade = line.split(',')
    grade = int(grade)
    if grade < 65:
        accumulator.add(1)
    return(student,course, grade)

# COMMAND ----------

# compute averages in spark
gradesRDD = sc.textFile(demo4_path+'grades.csv')\
              .map(lambda x: parse_grades(x, nFailing))
studentAvgs = gradesRDD.map(lambda x: (x[0], (x[2], 1)))\
                       .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))\
                       .mapValues(lambda x: x[0]/x[1])
courseAvgs = gradesRDD.map(lambda x: (x[1], (x[2], 1)))\
                      .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))\
                      .mapValues(lambda x: x[0]/x[1])

# COMMAND ----------

# take a look
print("===== average by student =====")
print(studentAvgs.collect())
print("===== average by course =====")
print(courseAvgs.collect())
print("===== number of failing grades awarded =====")
print(nFailing)

# COMMAND ----------

# MAGIC %md > __DISCUSSION QUESTIONS:__
# MAGIC * What is wrong with the results? (__`HINT:`__ _how many failing grades are there really?_)
# MAGIC * Why might this be happening? (__`HINT:`__ _How many actions are there in this code? Which parts of the DAG are recomputed for each of these actions?_)
# MAGIC * What one line could we add to the code to fix this problem?
# MAGIC   * What could go wrong with our "fix"?
# MAGIC * How could we have designed our parser differently to avoid this problem in the first place?

# COMMAND ----------

# MAGIC %md # Exercise 6. WordCount & Naive Bayes Reprise
# MAGIC 
# MAGIC We'll wrap up today's demo by revisiting two tasks from weeks 1-2. Compare each of these Spark implementations to the approach we took when performing the same task in Hadoop MapReduce.

# COMMAND ----------

# MAGIC %md ### a) Word Count in Spark

# COMMAND ----------

# load the data into Spark
aliceRDD = sc.textFile(ALICE_TXT)

# COMMAND ----------

# perform wordcount
result = aliceRDD.flatMap(lambda line: re.findall('[a-z]+', line.lower())) \
                 .map(lambda word: (word, 1)) \
                 .reduceByKey(lambda a, b: a + b)\
                 .cache()

# COMMAND ----------

# take a look at the top 10 (by alphabet)
result.takeOrdered(10)

# COMMAND ----------

# take a look at the top 10 (by count)
result.takeOrdered(10, key=lambda x: -x[1])

# COMMAND ----------

# what does Spark consider the 'top'?
result.top(10)

# COMMAND ----------

# MAGIC %md > __DICUSSION QUESTIONS:__
# MAGIC * Compare/contrast this implementation to our Hadoop Streaming approach.
# MAGIC * How many times does the data get shuffled?
# MAGIC * What local aggregation will spark do?
# MAGIC * What is the difference between `take()` and `top()` and `takeOrdered()`? Is one more or less efficient than the others? Compare these actions to the work we had to do to sort and subset with multiple reducers in Hadoop MapReduce?
# MAGIC * What would happen if we removed the `cache()` that follows the `reduceByKey()`? [__`Hint:`__ _this is kind of a trick question, but try rerunning the job & look at the Spark UI..._]

# COMMAND ----------

# MAGIC %md ### b) Naive Bayes in Spark
# MAGIC Implement the algorithm (training & inference)

# COMMAND ----------

def parse(doc):
    """
    Helper Function to parse documents.
    """
    docID, class_, subj, body = doc.lower().split('\t')
    return(class_, subj + " " + body)

# COMMAND ----------

def tokenize(class_, text):
    """
    Map text from a given class to word list with counts for each class. 
    """
    # get words                       
    words = re.findall(r'[a-z]+', text)
    # emit a count for each class (0,1 or 1,0)
    class_counts = [1,0] if class_ =='0' else [0,1]
    return[(word, class_counts) for word in words]

# COMMAND ----------

def NBtrain(dataRDD, smoothing = 1.0):
    """
    Function to train a Naive Bayes Model in Spark.
    Returns a dictionary.
    """
    # extract word counts
    docsRDD = dataRDD.map(parse)
    wordsRDD = docsRDD.flatMap(lambda x: tokenize(*x)).cache()\
                      .reduceByKey(lambda x,y: np.array(x) + np.array(y))\
                      .cache()
    # compute priors
    docTotals = docsRDD.countByKey()
    priors = np.array([docTotals['0'], docTotals['1']])
    priors = priors/sum(priors)
    
    # compute conditionals
    wordTotals = sc.broadcast(wordsRDD.map(lambda x: x[1] + np.array([smoothing, smoothing]))\
                                      .reduce(lambda x,y: np.array(x) + np.array(y)))
    cProb = wordsRDD.mapValues(lambda x: x + np.array([smoothing, smoothing]))\
                    .mapValues(lambda x: x/np.array(wordTotals.value))\
                    .collect()
    
    return dict([("ClassPriors", priors)] + cProb)

# COMMAND ----------

def NBclassify(document, model_dict):
    """
    Classify a document as ham/spam via Naive Bayes.
    Use logProbabilities to avoid floating point error.
    NOTE: this is just a python function, no distribution so 
    we should expect our documents (& model) to fit in memory.
    """
    # get words                       
    words = re.findall(r'[a-z]+', document.lower())
    # compute log probabilities
    logProbs = [np.log(model_dict.get(wrd,[1,1])) for wrd in words]
    # return most likely class
    sumLogProbs = np.log(model_dict['ClassPriors']) + sum(logProbs)
    return np.argmax(sumLogProbs)

# COMMAND ----------

def evaluate(resultsRDD):
    """
    Compute accuracy, precision, recall an F1 score given a
    pairRDD of (true_class, predicted_class)
    """
    nDocs = resultsRDD.count()
    TP = resultsRDD.filter(lambda x: x[0] == '1' and x[1] == 1).count()
    TN = resultsRDD.filter(lambda x: x[0] == '0' and x[1] == 0).count()
    FP = resultsRDD.filter(lambda x: x[0] == '0' and x[1] == 1).count()
    FN = resultsRDD.filter(lambda x: x[0] == '1' and x[1] == 0).count()
    
    # report results 
    print(f"Total # Documents:\t{nDocs}")
    print(f"True Positives:\t{TP}") 
    print(f"True Negatives:\t{TN}")
    print(f"False Positives:\t{FP}")
    print(f"False Negatives:\t{FN}") 
    print(f"Accuracy\t{(TP + TN)/(TP + TN + FP + FN)}")
    if (TP + FP) != 0:  
        precision = TP / (TP + FP)  
        print(f"Precision\t{precision}")
    if (TP + FN) != 0: 
        recall = TP / (TP + FN) 
        print(f"Recall\t{recall}") 
    if TP != 0: 
        f_score = 2 * precision * recall / (precision + recall)
        print(f"F-Score\t{f_score}")

# COMMAND ----------

# MAGIC %md Retrieve results.

# COMMAND ----------

# load data into Spark
trainRDD = sc.textFile(TRAIN_PATH)
testRDD = sc.textFile(TEST_PATH)

# COMMAND ----------

# train your model (& take a look)
NBmodel = NBtrain(trainRDD)
NBmodel

# COMMAND ----------

# perform inference on a doc (just to test)
NBclassify("This Japan Tokyo Macao is Chinese", NBmodel)

# COMMAND ----------

# evaluate your model
model_b = sc.broadcast(NBmodel)
resultsRDD = testRDD.map(parse)\
                    .mapValues(lambda x: NBclassify(x, model_b.value))
evaluate(resultsRDD)

# COMMAND ----------

# MAGIC %md > __DICUSSION QUESTIONS:__
# MAGIC * Compare/contrast this implementation to our Hadoop Streaming approach.

# COMMAND ----------

# MAGIC %md 

# COMMAND ----------

# MAGIC %md 

# COMMAND ----------



# COMMAND ----------

