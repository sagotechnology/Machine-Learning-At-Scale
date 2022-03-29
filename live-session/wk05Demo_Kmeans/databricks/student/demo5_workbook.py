# Databricks notebook source
# MAGIC %md # Demo 5 - Accumulators, Aggregations, and Joins in Spark
# MAGIC __`MIDS w261: Machine Learning at Scale | UC Berkeley School of Information | Spring 2019`__
# MAGIC 
# MAGIC 
# MAGIC By the end of this demo you should be able to: 
# MAGIC * ... __implement__ a custom accumulator
# MAGIC * ... __explain__ different types of aggregations and how they are implemented in Spark.
# MAGIC * ... __explain__ how different join operations are implemented in Spark
# MAGIC * ... __explain__  the challenges of implementing the A Priori algorithm at Scale

# COMMAND ----------

# MAGIC %md ### Notebook Set-Up

# COMMAND ----------

# MAGIC %md ### Run the next three cells to create your DEMO5 directory 
# MAGIC The scala code below fetches your username automatically and creates a temporary Spark table that can be read by python in the following cell. Don't worry about understanding this code. 

# COMMAND ----------

# RUN THIS CELL AS IS
# This code snippet reads the user directory name, and stores is in a python variable.
# Next, it creates a folder inside your home folder, which you will use for files which you save inside this notebook.
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
userhome = 'dbfs:/user/' + username
print(userhome)
demo5_path = userhome + "/demo5/" 
demo5_path_open = '/dbfs' + demo5_path.split(':')[-1] # for use with python open()
dbutils.fs.mkdirs(demo5_path)

# COMMAND ----------

# RUN THIS CELL AS IS
# Here we'll create a test file, and use databricks utils to makes usre everything works as expected.
# You should see a result like: dbfs:/user/<your email>@ischool.berkeley.edu/demo4/test.txt
dbutils.fs.put(demo5_path+'test5.txt',"hello world",True)
display(dbutils.fs.ls(demo5_path))

# COMMAND ----------

# imports
import sys
import numpy as np
import matplotlib.pyplot as plt

# COMMAND ----------

sc = spark.sparkContext

# COMMAND ----------

# MAGIC %md # Accumulators
# MAGIC Definitive Guide book, pg. 241

# COMMAND ----------

# MAGIC %md Accumulators are Spark's equivalent of Hadoop counters. Like broadcast variables they represent shared information across the nodes in your cluster, but unlike broadcast variables accumulators are _write-only_ ... in other words you can only access their values in the driver program and not on your executors (where transformations are applied). As convenient as this sounds, there are a few common pitfalls to avoid. Let's take a look.
# MAGIC 
# MAGIC Run the following cell to create a sample data file representing a list of `studentID, courseID, final_grade`...

# COMMAND ----------

# MAGIC %md ## Exercise 1

# COMMAND ----------

dbutils.fs.put(demo5_path+"grades.csv", 
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

# function to increment the accumulator as we read in the data
def parse_grades(line, accumulator):
    """Helper function to parse input & track failing grades."""
    student,course,grade = line.split(',')
    grade = int(grade)
    if grade < 65:
        accumulator.add(1)
    return(student,course, grade)

# COMMAND ----------

# initialize an accumulator to track failing grades
nFailing = sc.accumulator(0)

# COMMAND ----------

# compute averages in spark

gradesRDD = sc.textFile(demo5_path+'grades.csv')\
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

# MAGIC %md ## Custom Accumulators
# MAGIC https://spark.apache.org/docs/latest/rdd-programming-guide.html#accumulators

# COMMAND ----------

# MAGIC %md While SparkContext supports accumulators for primitive data types like int and float, users can also define accumulators for custom types by providing a custom AccumulatorParam object. 
# MAGIC 
# MAGIC We may want to utilize custom accumulators later in the course when we implement PageRank, or Shortest Path (graph) algorithms

# COMMAND ----------

from pyspark.accumulators import AccumulatorParam

# Spark only implements Accumulator parameter for numeric types.
# This class extends Accumulator support to the string type.
class StringAccumulatorParam(AccumulatorParam):
    def zero(self, value):
        return value
    def addInPlace(self, val1, val2):
        return val1 +" -> "+ val2

# COMMAND ----------

# MAGIC %md # Aggregations

# COMMAND ----------

# MAGIC %md ## groupByKey()
# MAGIC https://spark.apache.org/docs/latest/api/python/_modules/pyspark/rdd.html#RDD.groupByKey   
# MAGIC Easy to reason about, because it's very familiar folks coming from the SQL world. However, for the majority of cases, this is the wrong approach. The fundamental issue here is that each executor must hold all values for a given key in memory before applying the function to them.

# COMMAND ----------

displayHTML('<img src="https://github.com/kyleiwaniec/w261_assets/blob/master/images/groupbykey.png?raw=true"/>')

# COMMAND ----------

# MAGIC %md ## reduceByKey(Func)
# MAGIC https://spark.apache.org/docs/latest/api/python/_modules/pyspark/rdd.html#RDD.reduceByKey    
# MAGIC A much more stable approach to additive problems is reduceByKey. This is because the reduce happens within each partition and doesn’t need to put everything in memory. Additionally, there is no incurred shuffle during this operation; everything happens at each worker individually before performing the final reduce.

# COMMAND ----------

displayHTML('<img src="https://github.com/kyleiwaniec/w261_assets/blob/master/images/reducebykey.png?raw=true"/>')

# COMMAND ----------

# MAGIC %md ## combineByKey(createCombiner, mergeValue, mergeCombiners)
# MAGIC https://spark.apache.org/docs/latest/api/python/_modules/pyspark/rdd.html#RDD.combineByKey   
# MAGIC The first function input to the combiner specifies how to merge values, and the second function specifies how to merge combiners. For example, we might want to add values to a list, and subsequently merge the lists.

# COMMAND ----------

displayHTML('<img src="https://github.com/kyleiwaniec/w261_assets/blob/master/images/combinebykey.png?raw=true"/>')

# COMMAND ----------

# MAGIC %md ## foldByKey(zeroValue, Func)
# MAGIC https://spark.apache.org/docs/latest/api/python/_modules/pyspark/rdd.html#RDD.foldByKey   
# MAGIC Calls combineByKey, but allows us to use a zero value which can be added to the result an arbitrary number of
# MAGIC times, and must not change the result (eg. 0 for addition, 1 for multiplication)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## aggregateByKey(zeroValue, seqOp, combOp)
# MAGIC https://spark.apache.org/docs/latest/api/python/_modules/pyspark/rdd.html#RDD.aggregateByKey

# COMMAND ----------

displayHTML('<img src="https://github.com/kyleiwaniec/w261_assets/blob/master/images/aggregatebykey.png?raw=true"/>')

# COMMAND ----------

# MAGIC %md ## treeAggregate(zeroValue, seqOp, combOp, depth)
# MAGIC https://spark.apache.org/docs/latest/api/python/_modules/pyspark/rdd.html#RDD.treeAggregate    
# MAGIC Same as aggregate except it “pushes down” some of the subaggregations (creating a tree from executor to executor)
# MAGIC before performing nal aggregations on the driver.

# COMMAND ----------

# MAGIC %md ## Back to our example: 
# MAGIC What if we wanted to get a list of letter grades that each student recieved as well as their average?

# COMMAND ----------

# MAGIC %md 
# MAGIC Let's switch gears now for a moment, and look at aggregations. We'll come back to our string accumulator later in the following example.

# COMMAND ----------

# MAGIC %md ## Exercise 2

# COMMAND ----------

def toLetterGrade(x):
    if x > 92.0:
        return "A"
    elif x > 82.0:
        return "B"
    elif x > 72.0:
        return "C"
    elif x > 65.0:
        return "D"
    else:
        return "F"

def getCounts(a,b):
    return (a[0] + b[0], a[1] + b[1], toLetterGrade(a[0])+toLetterGrade(b[0]))
    
studentAvgs = gradesRDD.map(lambda x: (x[0], (x[2], 1)))\
                       .reduceByKey(getCounts)\
                       .mapValues(lambda x: ((x[0]/x[1]),x[2]))\
                       .collect()


# COMMAND ----------

# MAGIC %md ## How can we debug this problem?
# MAGIC 1. What is the error?
# MAGIC 2. Insert as many cells as you need to figure out what happened. Next we'll look at some ways to "fix" it.

# COMMAND ----------

# Your solutions here....

# COMMAND ----------

# MAGIC %md ## Let's look at some alternative implementations:

# COMMAND ----------

# MAGIC %md ### foldByKey allows us to specify a zero value

# COMMAND ----------

#gradesRDD.map(lambda x: (x[0], (x[2], 1))).foldByKey((0,0,""),getCounts).collect()

# COMMAND ----------

# MAGIC %md ### Can we solve this problem using a combineByKey which provides more granular control over the parameters
# MAGIC https://backtobazics.com/big-data/apache-spark-combinebykey-example/

# COMMAND ----------

def createCombiner(a):
    return a

def mergeValues(a,b):
    return (a[0] + b[0], a[1] + b[1], toLetterGrade(a[0])+toLetterGrade(b[0]));

def mergeCombiners(a,b):
    return (a[0] + b[0], a[1] + b[1], toLetterGrade(a[0])+toLetterGrade(b[0]))

studentAvgs = gradesRDD.map(lambda x: (x[0], (x[2], 1)))\
                       .combineByKey(createCombiner,mergeValues,mergeCombiners)\
                       .mapValues(lambda x: ((x[0]/x[1]),x[2]))

# COMMAND ----------

gradesRDD.map(lambda x: (x[0], (x[2], 1))).combineByKey(createCombiner,mergeValues,mergeCombiners).collect()

# COMMAND ----------

# MAGIC %md ### aggragateByKey 
# MAGIC aggragateByKey requires a null and start value as well as two different functions. One to aggregate within partitions, and one to aggregate across partitions

# COMMAND ----------

def seqOp(a,b):
    return(a[0] + b[0], a[1] + b[1], a[2]+toLetterGrade(b[2]))

def combOp(a,b):
    return (a+b);

# COMMAND ----------

gradesRDD.map(lambda x: (x[0], (x[2], 1, x[2])))\
         .aggregateByKey((0,0,""),seqOp,combOp)\
         .mapValues(lambda x: ((x[0]/x[1]),x[2]))\
         .collect()


# COMMAND ----------

letterAccum = sc.accumulator("===", StringAccumulatorParam())
gradesRDD.foreach(lambda x: letterAccum.add(toLetterGrade(x[2])))
print (letterAccum)

# COMMAND ----------

# MAGIC %md # Joins

# COMMAND ----------

# MAGIC %md * join       
# MAGIC https://spark.apache.org/docs/latest/api/python/_modules/pyspark/rdd.html#RDD.join     
# MAGIC * leftOuterJoin   
# MAGIC https://spark.apache.org/docs/latest/api/python/_modules/pyspark/rdd.html#RDD.leftOuterJoin    
# MAGIC * rightOuterJoin   
# MAGIC https://spark.apache.org/docs/latest/api/python/_modules/pyspark/rdd.html#RDD.rightOuterJoin    
# MAGIC * fullOuterJoin   
# MAGIC https://spark.apache.org/docs/latest/api/python/_modules/pyspark/rdd.html#RDD.fullOuterJoin   
# MAGIC * cartesian   
# MAGIC https://spark.apache.org/docs/latest/api/python/_modules/pyspark/rdd.html#RDD.cartesian

# COMMAND ----------

x = sc.parallelize([("a", 1), ("b", 4)])
y = sc.parallelize([("a", 2), ("c", 8)])
sorted(x.fullOuterJoin(y).collect())
#[('a', (1, 2)), ('b', (4, None)), ('c', (None, 8))]

# COMMAND ----------

sorted(x.rightOuterJoin(y).collect())

# COMMAND ----------

sorted(x.leftOuterJoin(y).collect())

# COMMAND ----------

# MAGIC %md Lets load some data for the code examples

# COMMAND ----------

person = spark.createDataFrame([
    (0, "Bill Chambers", 0, [100]),
    (1, "Matei Zaharia", 1, [500, 250, 100]),
    (2, "Michael Armbrust", 1, [250, 100])])\
  .toDF("id", "name", "graduate_program", "spark_status")
graduateProgram = spark.createDataFrame([
    (0, "Masters", "School of Information", "UC Berkeley"),
    (2, "Masters", "EECS", "UC Berkeley"),
    (1, "Ph.D.", "EECS", "UC Berkeley")])\
  .toDF("id", "degree", "department", "school")
sparkStatus = spark.createDataFrame([
    (500, "Vice President"),
    (250, "PMC Member"),
    (100, "Contributor")])\
  .toDF("id", "status")

# COMMAND ----------

# run as is
joinExpression = person["graduate_program"] == graduateProgram['id']

# COMMAND ----------

# run as is
wrongJoinExpression = person["name"] == graduateProgram["school"]

# COMMAND ----------

# run as is
person.join(graduateProgram, joinExpression).show()

# COMMAND ----------

person.join(graduateProgram, wrongJoinExpression).show()

# COMMAND ----------

# Spark perfoms an "inner" join by default. But we can specify this explicitly.
# Try different join types.
joinType = "outer"
joinType = "left_outer"
joinType = "right_outer"

# COMMAND ----------

person.join(graduateProgram, joinExpression, joinType).show()

# COMMAND ----------

# MAGIC %md ### Which keys do outer joins evaluate?

# COMMAND ----------

# MAGIC %md ### A departure from traditional joins:

# COMMAND ----------

gradProgram2 = graduateProgram.union(spark.createDataFrame([
    (0, "Masters", "Duplicated Row", "Duplicated School")]))
gradProgram2.createOrReplaceTempView("gradProgram2")

# COMMAND ----------

# Think of left semi joins as filters on a DataFrame, as opposed to the function of a conventional join
joinType = "left_semi"
gradProgram2.join(person, joinExpression, joinType).show()

# COMMAND ----------

gradProgram2.show()

# COMMAND ----------

joinType = "left_anti"
gradProgram2.join(person, joinExpression, joinType).show()

# COMMAND ----------

# MAGIC %md ### Natural Joins
# MAGIC 
# MAGIC __DANGER__: Natural joins make implicit guesses at the columns on which you would like to join. Why is this bad?

# COMMAND ----------

# MAGIC %md ### Cross (Cartesian) Joins
# MAGIC Or, Cartesian products. Cross joinsare inner joins that do not specify a predicate. Cross joins will join every single row in the left DataFrame with every single row in the right DataFrame

# COMMAND ----------

joinType = "cross"
graduateProgram.join(person, joinExpression, joinType).show()

# COMMAND ----------

person.crossJoin(graduateProgram).show()

# COMMAND ----------

# MAGIC %md __DANGER__: How many rows would we end up with from a cross join if each table had 1000 rows?