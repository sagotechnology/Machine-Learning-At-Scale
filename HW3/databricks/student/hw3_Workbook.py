# Databricks notebook source
# MAGIC %md # HW 3 - Synonym Detection In Spark
# MAGIC __`MIDS w261: Machine Learning at Scale | UC Berkeley School of Information | Fall 2018`__
# MAGIC 
# MAGIC In the last homework assignment you performed Naive Bayes to classify documents as 'ham' or 'spam.' In doing so, we relied on the implicit assumption that the list of words in a document can tell us something about the nature of that document's content. We'll rely on a similar intuition this week: the idea that, if we analyze a large enough corpus of text, the list of words that appear in small window before or after a vocabulary term can tell us something about that term's meaning. This is similar to the intuition behind the word2vec algorithm.
# MAGIC 
# MAGIC This will be your first assignment working in Spark. You'll perform Synonym Detection by repurposing an algorithm commonly used in Natural Language Processing to perform document similarity analysis. In doing so you'll also become familiar with important datatypes for efficiently processing sparse vectors and a number of set similarity metrics (e.g. Cosine, Jaccard, Dice). By the end of this homework you should be able to:  
# MAGIC * ... __define__ the terms `one-hot encoding`, `co-occurrance matrix`, `stripe`, `inverted index`, `postings`, and `basis vocabulary` in the context of both synonym detection and document similarity analysis.
# MAGIC * ... __explain__ the reasoning behind using a word stripe to compare word meanings.
# MAGIC * ... __identify__ what makes set-similarity calculations computationally challenging.
# MAGIC * ... __implement__ stateless algorithms in Spark to build stripes, inverted index and compute similarity metrics.
# MAGIC * ... __identify__ when it makes sense to take a stripe approach and when to use pairs
# MAGIC * ... __apply__ appropriate metrics to assess the performance of your synonym detection algorithm. 
# MAGIC 
# MAGIC __RECOMMENDED READING FOR HW3__:	
# MAGIC Your reading assignment for weeks 4 and 5 were fairly heavy and you may have glossed over the papers on dimension independent similarity metrics by [Zadeh et al](http://stanford.edu/~rezab/papers/disco.pdf) and pairwise document similarity by [Elsayed et al](https://terpconnect.umd.edu/~oard/pdf/acl08elsayed2.pdf). If you haven't already, this would be a good time to review those readings, especially when it comes to the similarity formulas -- they are directly relevant to this assignment.
# MAGIC 
# MAGIC DITP Chapter 4 - Inverted Indexing for Text Retrieval. While this text is specific to Hadoop, the Map/Reduce concepts still apply.
# MAGIC 
# MAGIC __Please refer to the `README` for homework submission instructions and additional resources.__

# COMMAND ----------

# MAGIC %md # Notebook Set-Up
# MAGIC Before starting your homework run the following cells to confirm your setup.

# COMMAND ----------

import re
import ast
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md ### Run the next cell to create your directory in dbfs
# MAGIC You do not need to understand this scala snippet. It simply dynamically fetches your user directory name so that any files you write can be saved in your own directory.

# COMMAND ----------

# RUN THIS CELL AS IS
# This code snippet reads the user directory name, and stores is in a python variable.
# Next, it creates a folder inside your home folder, which you will use for files which you save inside this notebook.
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
userhome = 'dbfs:/user/' + username
print(userhome)
hw3_path = userhome + "/hw3/" 
hw3_path_open = '/dbfs' + hw3_path.split(':')[-1] # for use with python open()
dbutils.fs.mkdirs(hw3_path)

# COMMAND ----------

# RUN THIS CELL AS IS. You should see multiple google-eng-all-5gram-* files in the results. If you do not see these, please let an Instructor or TA know.
display(dbutils.fs.ls('/mnt/mids-w261/HW3/'))

# COMMAND ----------

# RUN THIS CELL AS IS - A test to make sure your directory is working as expected.
# You should see a result like:
# dbfs:/user/youremail@ischool.berkeley.edu/hw3/sample_docs.txt
dbutils.fs.put(hw3_path+'test.txt',"hello world",True)
display(dbutils.fs.ls(hw3_path))

# COMMAND ----------

# get Spark Session info (RUN THIS CELL AS IS)
spark

# COMMAND ----------

# start SparkContext (RUN THIS CELL AS IS)
sc = spark.sparkContext

# COMMAND ----------

# Spark configuration Information (RUN THIS CELL AS IS)
# sc.getConf().getAll()

# COMMAND ----------

# MAGIC %md # Question 1: Spark Basics.
# MAGIC In your readings and live session demos for weeks 4 and 5 you got a crash course in working with Spark. We also talked about how Spark RDDs fit into the broader picture of distributed algorithm design. The questions below cover key points from these discussions. Answer each one very briefly - 2 to 3 sentences.
# MAGIC 
# MAGIC ### Q1 Tasks:
# MAGIC 
# MAGIC * __a) short response:__ What is Spark? How  does it relate to Hadoop MapReduce?
# MAGIC 
# MAGIC * __b) short response:__ In what ways does Spark follow the principles of statelessness (a.k.a. functional programming)? List at least one way in which it allows the programmer to depart from this principle. 
# MAGIC 
# MAGIC * __c) short response:__ In the context of Spark what is a 'DAG' and how does it relate to the difference between an 'action' and a 'transformation'? Why is it useful to pay attention to the DAG that underlies your Spark implementation?

# COMMAND ----------

# MAGIC %md ### Q1 Student Answers:
# MAGIC > __a)__ Type your answer here!
# MAGIC 
# MAGIC > __b)__ Type your answer here!
# MAGIC 
# MAGIC > __c)__ Type your answer here!  

# COMMAND ----------

# MAGIC %md # Question 2: Similarity Metrics
# MAGIC As mentioned in the introduction to this assignment, an intuitive way to compare the meaning of two documents is to compare the list of words they contain. Given a vocabulary \\(V\\) (feature set) we would represent each document as a vector of `1`-s and `0`-s based on whether or not it contains each word in \\(V\\). These "one-hot encoded" vector representations allow us to use math to identify similar documents. However like many NLP tasks the high-dimensionality of the feature space is a challenge... especially when we start to scale up the size and number of documents we want to compare.
# MAGIC 
# MAGIC In this question we'll look at a toy example of document similarity analysis. Consider these 3 'documents': 
# MAGIC ```
# MAGIC docA	the flight of a bumblebee
# MAGIC docB	the length of a flight
# MAGIC docC	buzzing bumblebee flight
# MAGIC ```
# MAGIC These documents have a total of \\(7\\) unique words: 
# MAGIC >`a, bumblebee, buzzing, flight, length, of, the`.     
# MAGIC 
# MAGIC Given this vocabulary, the documents' vector representations are (note that one-hot encoded entries follow the order of the vocab list above):
# MAGIC 
# MAGIC ```
# MAGIC docA	[1,1,0,1,0,1,1]
# MAGIC docB	[1,0,0,1,1,1,1]
# MAGIC docC	[0,1,1,1,0,0,0]
# MAGIC ```  
# MAGIC 
# MAGIC ### Q2 Tasks:
# MAGIC 
# MAGIC * __a) short response:__ The cosine similarity between two vectors is $$\frac{A\cdot B}{\|A\|\|B\|}$$. Explain what the the numerator and denominator of this calculation would represent in terms of word counts in documents A and B. 
# MAGIC 
# MAGIC * __b) short response:__ Explain how the Jaccard, Overlap and Dice metrics are similar/different to the calculation for cosine similarity. When would these metrics lead to different similarity rankings for a set of documents? HINT: consider documents of very different lengths. It may be helpful to generate some small examples.
# MAGIC 
# MAGIC * __c) short response:__ Calculate the cosine similarity for each pair of documents in our toy corpus. Please use markdown and \\(\LaTeX\\) to show your calcuations.  
# MAGIC 
# MAGIC * __d) short response:__ According to your calculations in `part c` which pair of documents are most similar in meaning? __BONUS__: Does this match your expecatation from reading the documents? If not, speculate about why we might have gotten this result.
# MAGIC 
# MAGIC * __e) short response:__ In NLP common words like '`the`', '`of`', and '`a`' increase our feature space without adding a lot of signal about _semantic meaning_. Repeat your analysis from `part c` but this time ignore these three words in your calculations [__`TIP:`__ _to 'remove' stopwords just ignore the vector entries in columns corresponding to the words you wish to disregard_]. How do your results change?

# COMMAND ----------

# MAGIC %md ### Q2 Student Answers:
# MAGIC > __a)__ Type your answer here!
# MAGIC 
# MAGIC > __b)__ Type your answer here!
# MAGIC 
# MAGIC > __c)__ Type your answer here!  
# MAGIC 
# MAGIC > __d)__ Type your answer here!    
# MAGIC > __d-bonus)__ Type your answer here!
# MAGIC 
# MAGIC > __e)__ Type your answer here!

# COMMAND ----------

# MAGIC %md # Question 3: Synonym Detection Strategy
# MAGIC 
# MAGIC In the Synonym Detection task we want to compare the meaning of words, not documents. For clarity, lets call the words whose meaning we want to compare `terms`. If only we had a 'meaning document' for each `term` then we could easily use the document similarity strategy from Question 2 to figure out which `terms` have similar meaning (i.e. are 'synonyms'). Of course in order for that to work we'd have to reasonably believe that the words in these 'meaning documents' really do reflect the meaning of the `term`. For a good analysis we'd also need these 'meaning documents' to be fairly long -- the one or two sentence dictionary definition of a term isn't going to provide enough signal to distinguish between thousands and thousands of `term` meanings.
# MAGIC 
# MAGIC This is where the idea of co-occurrance comes in. Just like DocSim makes the assumption that words in a document tell us about the document's meaning, we're going to assume that the set of words that 'co-occur' within a small window around our term can tell us some thing about the meaning of that `term`. Remember that we're going to make this 'co-words' list (a.k.a. 'stripe') by looking at a large body of text. This stripe is our 'meaning document' in that it reflects all the kinds of situations in which our `term` gets used in real language. So another way to phrase our assumption is: we think `terms` that get used to complete lots of the same phrases probably have related meanings. This may seem like an odd assumption but computational linguists have found that it works surprisingly well in practice. Let's look at a toy example to build your intuition for why and how.
# MAGIC 
# MAGIC Consider the opening line of Charles Dickens' _A Tale of Two Cities_:

# COMMAND ----------

corpus = """It was the best of times, it was the worst of times, 
it was the age of wisdom it was the age of foolishness"""

# COMMAND ----------

# MAGIC %md There are a total of 10 unique words in this short 'corpus':

# COMMAND ----------

words = list(set(re.findall(r'\w+', corpus.lower())))
print(words)

# COMMAND ----------

# MAGIC %md But of these 10 words, 4 are so common that they probably don't tell us very much about meaning.

# COMMAND ----------

stopwords = ["it", "the", "was", "of"]

# COMMAND ----------

# MAGIC %md So we'll ignore these 'stop words' and we're left with a 6 word vocabulary:

# COMMAND ----------

vocab = sorted([w for w in words if w not in stopwords])
print(vocab)

# COMMAND ----------

# MAGIC %md Your goal in the tasks below is to asses, which of these six words are most related to each other in meaning -- based solely on this short two line body of text.
# MAGIC 
# MAGIC ### Q3 Tasks:
# MAGIC 
# MAGIC * __a) short response:__ Given this six word vocabulary, how many 'pairs' of words do we want to compare? More generally for a n-word vocabulary how many pairwise comparisons are there to make? 
# MAGIC 
# MAGIC * __b) code:__  In the space provided below, create a 'stripe' for each term in the vocabulary. This stripe should be the list of all other vocabulary words that occur within a 5 word window (two words on either side) of the term's position in the original text (In this exercise, use ['it', 'was', 'the','of'] as stopwords, just ignore them from your 5 word vectors).
# MAGIC 
# MAGIC * __c) short response:__ Run the provided code to turn your stripes into a 1-hot encoded co-occurrence matrix. For our 6 word vocabulary how many entries are in this matrix? How many entries are zeros? 
# MAGIC 
# MAGIC * __d) code:__ Complete the provided code to loop over all pairs and compute their cosine similarity. Please do not modify the existing code, just add your own in the spot marked.
# MAGIC 
# MAGIC * __e) short response:__ Which pairs of words have the highest 'similarity' scores? __BONUS__: Are these words 'synonyms' in the traditional sense? In what sense are their meanings 'similar'? Explain how our results are contingent on the input text. What would change if we had a much larger corpus?

# COMMAND ----------

# MAGIC %md ### Q3 Student Answers:
# MAGIC > __a)__ Type your answer here!
# MAGIC 
# MAGIC > __c)__ Type your answer here!   
# MAGIC 
# MAGIC > __e)__ Type your answer here!   
# MAGIC __e-bonus)__ Type your answer here!

# COMMAND ----------

# for convenience, here are the corpus & vocab list again (RUN THIS CELL AS IS)
print("CORPUS:")
print(corpus)
print('VOCAB:')
print(vocab)

# COMMAND ----------

# MAGIC %md
# MAGIC <img src='https://github.com/kyleiwaniec/w261_assets/blob/master/images/best-of-times.png?raw=true' style='width:80%'>

# COMMAND ----------

# part b - USE THE TEXT ABOVE TO COMPLETE EACH STRIPE
# Stopwords: 
#     ['it', 'was', 'the', 'of'] 
# Hint:
#     In provided sentence, age appears in two 5 word vectors: ['was', 'the', 'age', 'of', 'wisdom'] and ['was', 'the', 'age', 'of', 'foolishness']
#     After removing stopwords, the remaining words are 'wisdom' and 'foolishness'
#
#     You finish the rest of the non-stopwords below. 

stripes = {'age':['wisdom','foolishness'], # example
           'best':[], # YOU FILL IN THE REST
           'foolishness':[],
           'times': [],
           'wisdom':[],
           'worst':[]}

# COMMAND ----------

# part c - initializing an empty co-occurrence matrix (RUN THIS CELL AS IS)
co_matrix = pd.DataFrame({term: [0]*len(vocab) for term in vocab}, index = vocab, dtype=int)

# COMMAND ----------

# part c - this cell 1-hot encodes the co-occurrence matrix (RUN THIS CELL AS IS) 
for term, nbrs in stripes.items():
    pass
    for nbr in nbrs:
        co_matrix.loc[term, nbr] = 1
co_matrix

# COMMAND ----------

# part e - FILL IN THE MISSING LINES to compute the cosine similarity between each pair of terms
for term1, term2 in itertools.combinations(vocab, 2):
    # one hot-encoded vectors
    v1 = co_matrix[term1]
    v2 = co_matrix[term2]
    
    # cosine similarity
    ############# YOUR CODE HERE #################
    csim = None
    ############# (END) YOUR CODE #################    
    
    print(f"{term1}-{term2}: {csim}")

# COMMAND ----------

# MAGIC %md # Question 4: Pairs and Stripes at Scale
# MAGIC 
# MAGIC As you read in the paper by Zadeh et al, the advantage of metrics like Cosine, Dice, Overlap and Jaccard is that they are dimension independent -- that is to say, if we implement them in a smart way the computational complexity of performing these computations is independent of the number of documents we want to compare (or in our case, the number of terms that are potential synonyms). One component of a 'smart implementation' involves thinking carefully both about how you define the "basis vocabulary" that forms your feature set (removing stopwords, etc). Another key idea is to use a data structure that facilitates distributed calculations. The DISCO implemetation further uses a sampling strategy, but that is beyond the scope of this assignment. 
# MAGIC 
# MAGIC In this question we'll take a closer look at the computational complexity of the synonym detection approach we took in question 3 and then revist the document similarity example as a way to explore a more efficient approach to parallelizing this analysis.
# MAGIC 
# MAGIC ### Q4 Tasks:
# MAGIC 
# MAGIC * __a) short response:__ In question 3 you calculated the cosine similarity of pairs of words using the vector representation of their co-occurrences in a corpus. In the asynch videos about "Pairs and Stripes" you were introduced to an alternative strategy. Explain two ways that using these data structures are more efficient than 1-hot encoded vectors when it comes to distributed similarity calculations [__`HINT:`__ _Consider memory constraints, amount of information being shuffled, amount of information being transfered over the network, and level of parallelization._]
# MAGIC 
# MAGIC * __b) read provided code:__ The code below provides a streamined implementation of Document similarity analysis in Spark. Read through this code carefully. Once you are confident you understand how it works, answer the remaining questions. [__`TIP:`__ _to see the output of each transformation try commenting out the subsequent lines and adding an early `collect()` action_.]
# MAGIC 
# MAGIC * __c) short response:__ The second mapper function, `splitWords`, emits 'postings'. The list of all 'postings' for a word is also refered to as an 'inverted index'. In your own words, define each of these terms ('postings' and 'inverted index') based on your reading of the provided code. (*DITP by Lin and Dyer also contains a chapter on the Inverted Index although in the context of Hadoop rather than Spark. You may find the illustration in Chaprter 4 helpful in answering this question*).
# MAGIC 
# MAGIC * __d) short response:__ The third mapper, `makeCompositeKeys`, loops over the inverted index to emit 'pairs' of what? Explain what information is included in the composite key created at this stage and why it makes sense to synchronize around that information in the context of performing document similarity calculations. In addition to the information included in these new keys, what other piece of information will we need to compute Jaccard or Cosine similarity?
# MAGIC 
# MAGIC * __f) short response:__ Out of all the Spark transformations we make in this analysis, which are 'wide' transformations and which are 'narrow' transformations. Explain.

# COMMAND ----------

# MAGIC %md ### Q4 Student Answers:
# MAGIC 
# MAGIC > __a)__ Type your answer here! 
# MAGIC 
# MAGIC > __b)__ _read provided code before answering d-f_ 
# MAGIC 
# MAGIC > __c)__ Type your answer here!  
# MAGIC 
# MAGIC > __d)__ Type your answer here!  
# MAGIC 
# MAGIC > __e)__ Type your answer here!

# COMMAND ----------

# MAGIC %md A small test file: __`sample_docs.txt`__

# COMMAND ----------

dbutils.fs.put(hw3_path+"sample_docs.txt", 
"""docA	bright blue butterfly forget
docB	best forget bright sky
docC	blue sky bright sun
docD	under butterfly sky hangs
docE	forget blue butterfly""", True)

# COMMAND ----------

print(dbutils.fs.head(hw3_path+"sample_docs.txt"))

# COMMAND ----------

# MAGIC %md __Document Similarity Analysis in Spark:__

# COMMAND ----------

# load data - RUN THIS CELL AS IS
data = sc.textFile(hw3_path+"sample_docs.txt")  

# COMMAND ----------

# helper function - RUN THIS CELL AS IS
def splitWords(pair):
    """Mapper 2: tokenize each document and emit postings."""
    doc, text = pair
    words = text.split(" ")
    for w in words:
        yield (w, [(doc,len(words))])

# COMMAND ----------

# helper function - RUN THIS CELL AS IS
def makeCompositeKey(inverted_index):
    """Mapper 3: loop over postings and yield pairs."""
    word, postings = inverted_index
    # taking advantage of symmetry, output only (a,b), but not (b,a)
    for subset in itertools.combinations(sorted(postings), 2):
        yield (str(subset), 1)

# COMMAND ----------

# helper function - RUN THIS CELL AS IS
def jaccard(line):
    """Mapper 4: compute similarity scores"""
    (doc1, n1), (doc2, n2) = ast.literal_eval(line[0])
    total = int(line[1])
    jaccard = total / float(int(n1) + int(n2) - total)
    yield doc1+" - "+doc2, jaccard

# COMMAND ----------

# Spark Job - RUN THIS CELL AS IS
result = data.map(lambda line: line.split('\t')) \
             .flatMap(splitWords) \
             .reduceByKey(lambda x,y : x+y) \
             .flatMap(makeCompositeKey) \
             .reduceByKey(lambda x,y : x+y) \
             .flatMap(jaccard) \
             .takeOrdered(10, key=lambda x: -x[1])
result

# COMMAND ----------

# MAGIC %md # About the Data
# MAGIC Now that you are comfortable with similarity metrics we turn to the main task in this assignment: "Synonym" Detection. As you saw in Question 3 the ability of our algorithm to detect words with similar meanings is highly dependent on our input text. Specifically, we need a large enough corpus of natural language that we can expose our algorithm to a realistic range of contexts in which any given word might get used. Ideally, these 'contexts' would also provide enough signal to distinguish between words with similar semantic roles but different meaning. Finding such a corpus will be easier to accomplish for some words than others.
# MAGIC 
# MAGIC For the main task in this portion of the homework you will use data from Google's n-gram corpus. This data is particularly convenient for our task because Google has already done the first step for us: they windowed over a large subset of the web and extracted all 5-grams. If you are interested in learning more about this dataset the original source is: http://books.google.com/ngrams/, and a large subset is available [here from AWS](https://aws.amazon.com/datasets/google-books-ngrams/). 
# MAGIC 
# MAGIC For this assignment we have provided a subset of the 5-grams data consisting of 191 files of approximately 10MB each. These files are available in dbfs. Please only use the provided data so that we can ensure consistent results from student to student.
# MAGIC 
# MAGIC Each row in our dataset represents one of these 5 grams in the format:
# MAGIC > `(ngram) \t (count) \t (pages_count) \t (books_count)`
# MAGIC 
# MAGIC __DISCLAIMER__: In real life, we would calculate the stripes cooccurrence data from the raw text by windowing over the raw text and not from the 5-gram preprocessed data.  Calculating pairs on this 5-gram is a little corrupt as we will be double counting cooccurences. Having said that this exercise can still pull out some similar terms.

# COMMAND ----------

# RUN THIS CELL AS IS. You should see multiple google-eng-all-5gram-* files in the results. If you do not see these, please let an Instructor or TA know.
display(dbutils.fs.ls('/mnt/mids-w261/HW3/'))

# COMMAND ----------

# set global paths to full data folder and to the first file (which we'll use for testing)
NGRAMS = '/mnt/mids-w261/HW3'
F1_PATH = '/mnt/mids-w261/HW3/googlebooks-eng-all-5gram-20090715-0-filtered.txt'

# COMMAND ----------

# MAGIC %md As you develop your code you should use the following file to systems test each of your solutions before running it on the Google data. (Note: these are the 5-grams extracted from our two line Dickens corpus in Question 3... you should find that your Spark job results match the calculations we did "by hand").
# MAGIC 
# MAGIC Test file: __`systems_test.txt`__

# COMMAND ----------

dbutils.fs.put(hw3_path+"systems_test.txt",
"""it was the best of	1	1	1
age of wisdom it was	1	1	1
best of times it was	1	1	1
it was the age of	2	1	1
it was the worst of	1	1	1
of times it was the	2	1	1
of wisdom it was the	1	1	1
the age of wisdom it	1	1	1
the best of times it	1	1	1
the worst of times it	1	1	1
times it was the age	1	1	1
times it was the worst	1	1	1
was the age of wisdom	1	1	1
was the best of times	1	1	1
was the age of foolishness	1	1	1
was the worst of times	1	1	1
wisdom it was the age	1	1	1
worst of times it was	1	1	1""",True)

# COMMAND ----------

# MAGIC %md Finally, we'll create a Spark RDD for each of these files so that they're easy to access throughout the rest of the assignment.

# COMMAND ----------

# Spark RDDs for each dataset
testRDD = sc.textFile(hw3_path+"systems_test.txt") 
f1RDD = sc.textFile(F1_PATH)
dataRDD = sc.textFile(NGRAMS)

# COMMAND ----------

# MAGIC %md Let's take a peak at what each of these RDDs looks like:

# COMMAND ----------

testRDD.take(10)

# COMMAND ----------

f1RDD.take(10)

# COMMAND ----------

dataRDD.take(10)

# COMMAND ----------

# MAGIC %md # Question 5: N-gram EDA part 1 (words)
# MAGIC 
# MAGIC Before starting our synonym-detection, let's get a sense for this data. As you saw in questions 3 and 4 the size of the vocabulary will impact the amount of computation we have to do. Write a Spark job that will accomplish the three tasks below as efficiently as possible. (No credit will be awarded for jobs that sort or subset after calling `collect()`-- use the framework to get the minimum information requested). As you develop your code, systems test each job on the provided file with Dickens ngrams, then on a single file from the Ngram dataset before running the full analysis.
# MAGIC 
# MAGIC 
# MAGIC ### Q5 Tasks:
# MAGIC * __a) code:__ Write a Spark application to retrieve:
# MAGIC   * The number of unique words that appear in the data. (i.e. size of the vocabulary) 
# MAGIC   * A list of the top 10 words & their counts.
# MAGIC   * A list of the bottom 10 words & their counts.  
# MAGIC   
# MAGIC   __`NOTE  1:`__ _don't forget to lower case the ngrams before extracting words._  
# MAGIC   __`NOTE  2:`__ _don't forget to take in to account the number of occurances (count) of each ngram._  
# MAGIC   __`NOTE  3:`__ _to make this code more reusable, the `EDA1` function code base uses a parameter 'n' to specify the number of top/bottom words to print (in this case we've requested 10)._
# MAGIC 
# MAGIC 
# MAGIC * __b) short response:__ Given the vocab size you found in part a, how many potential synonym pairs could we form from this corpus? If each term's stripe were 1000 words long, how many tuples would we need to shuffle in order to form the inverted indices? Show and briefly explain your calculations for each part of this question. [__`HINT:`__ see your work from q4 for a review of these concepts.]
# MAGIC 
# MAGIC * __c) short response:__ Looking at the most frequent words and their counts, how usefull will these top words be in synonym detection? Explain.
# MAGIC 
# MAGIC * __d) short response:__ Looking at the least frequent words and their counts, how reliable should we expect the detected 'synonyms' for these words to be? Explain.

# COMMAND ----------

# MAGIC %md ### Q5 Student Answers:
# MAGIC 
# MAGIC > __b)__ Type your answer here!   
# MAGIC 
# MAGIC > __c)__ Type your answer here!   
# MAGIC 
# MAGIC > __d)__ Type your answer here!

# COMMAND ----------

# part a - write your spark job here 
def EDA1(rdd, n):
    total, top_n, bottom_n = None, None, None
    ############# YOUR CODE HERE ###############

    
    
    
    
    
    
    ############# (END) YOUR CODE ##############
    return total, top_n, bottom_n

# COMMAND ----------

# part a - run the system test (RUN THIS CELL AS IS... use display cell below to see results)
import time
start = time.time()
vocab_size, most_frequent, least_frequent = EDA1(testRDD, 10)
print("Wall time: {} seconds".format(time.time() - start))
# Command took 3.52 seconds -- by kylehamilton@ischool.berkeley.edu at 9/19/2020, 4:50:53 PM on kyle_hw

# COMMAND ----------

# part a - display results (feel free to modify the formatting code if needed)
print("Vocabulary Size:", vocab_size)
print(" ---- Top Words ----|--- Bottom Words ----")
for (w1, c1), (w2, c2) in zip(most_frequent, least_frequent):
    print(f"{w1:>8} {c1:>10} |{w2:>15} {c2:>3}")

# COMMAND ----------

# MAGIC %md Expected output for testRDD:
# MAGIC <pre>
# MAGIC     Vocabulary Size: 10
# MAGIC  ---- Top Words ----|--- Bottom Words ----
# MAGIC      was         17 |    foolishness   1
# MAGIC       of         17 |           best   4
# MAGIC      the         17 |          worst   5
# MAGIC       it         16 |         wisdom   5
# MAGIC    times         10 |            age   8
# MAGIC      age          8 |          times  10
# MAGIC    worst          5 |             it  16
# MAGIC   wisdom          5 |            was  17
# MAGIC     best          4 |             of  17
# MAGIC foolishness       1 |            the  17  
# MAGIC </pre>

# COMMAND ----------

# part a - run a single file, ie., a small sample (RUN THIS CELL AS IS)
start = time.time()
vocab_size, most_frequent, least_frequent = EDA1(f1RDD, 10)
print("Wall time: {} seconds".format(time.time() - start))
# Command took 4.93 seconds -- by kylehamilton@ischool.berkeley.edu at 9/19/2020, 4:51:04 PM on kyle_hw

# COMMAND ----------

# part a - display results (feel free to modify the formatting code if needed)
print("Vocabulary Size:", vocab_size)
print(" ---- Top Words ----|--- Bottom Words ----")
for (w1, c1), (w2, c2) in zip(most_frequent, least_frequent):
    print(f"{w1:>8} {c1:>10} |{w2:>15} {c2:>3}")

# COMMAND ----------

# MAGIC %md Expected output for f1RDD
# MAGIC <pre>
# MAGIC Vocabulary Size: 36353
# MAGIC  ---- Top Words ----|--- Bottom Words ----
# MAGIC      the   27691943 |    stakeholder  40
# MAGIC       of   18590950 |          kenny  40
# MAGIC       to   11601757 |         barnes  40
# MAGIC       in    7470912 |         arnall  40
# MAGIC        a    6926743 |     buonaparte  40
# MAGIC      and    6150529 |       puzzling  40
# MAGIC     that    4077421 |             hd  40
# MAGIC       is    4074864 |        corisca  40
# MAGIC       be    3720812 |       cristina  40
# MAGIC      was    2492074 |         durban  40
# MAGIC </pre>

# COMMAND ----------

# part a - run full analysis (RUN THIS CELL AS IS)
start = time.time()
vocab_size, most_frequent, least_frequent = EDA1(dataRDD, 10)
print("Wall time: {} seconds".format(time.time() - start))
# Command took 1.30 minutes -- by kylehamilton@ischool.berkeley.edu at 9/19/2020, 4:51:15 PM on kyle_hw

# COMMAND ----------

# part a - display results (feel free to modify the formatting code if needed)
print("Vocabulary Size:", vocab_size)
print(" ---- Top Words ----|--- Bottom Words ----")
for (w1, c1), (w2, c2) in zip(most_frequent, least_frequent):
    print(f"{w1:>8} {c1:>10} |{w2:>15} {c2:>3}")

# COMMAND ----------

# MAGIC %md Expected output for dataRDD:
# MAGIC (bottom words might vary a little due to ties)
# MAGIC <pre>
# MAGIC Vocabulary Size: 269339
# MAGIC  ---- Top Words ----|--- Bottom Words ----
# MAGIC      the 5490815394 |   schwetzingen  40
# MAGIC       of 3698583299 |           cras  40
# MAGIC       to 2227866570 |       parcival  40
# MAGIC       in 1421312776 |          porti  40
# MAGIC        a 1361123022 |    scribbler's  40
# MAGIC      and 1149577477 |      washermen  40
# MAGIC     that  802921147 |    viscerating  40
# MAGIC       is  758328796 |         mildes  40
# MAGIC       be  688707130 |      scholared  40
# MAGIC       as  492170314 |       jaworski  40
# MAGIC </pre>

# COMMAND ----------

# MAGIC %md # Question 6: N-gram EDA part 2 (co-occurrences)
# MAGIC 
# MAGIC The computational complexity of synonym analysis depends not only on the number of words, but also on the number of co-ocurrences each word has. In this question you'll take a closer look at that aspect of our data. As before, please test each job on small "systems test" (Dickens ngrams) file and on a single file from the Ngram dataset before running the full analysis.
# MAGIC 
# MAGIC ### Q6 Tasks:
# MAGIC * __a) code:__ Write a spark job that computes:
# MAGIC   * the number of unique neighbors (i.e. 5-gram co-occuring words) for each word in the vocabulary. 
# MAGIC   
# MAGIC  <pre>
# MAGIC   HINT: consider all words within a five-gram to be co-occuring. In other words, a word in a single 5-gram will always have 4 neighbors
# MAGIC   EXAMPLE:
# MAGIC     the dog ate cat litter 
# MAGIC     the cat has clean litter 
# MAGIC     
# MAGIC     Vocabulary:
# MAGIC     the, dog, ate, litter, cat, has, clean
# MAGIC     
# MAGIC     Neighbors:
# MAGIC     (the, dog) (the, ate) (the, cat) (the, littler), (dog, ate) (dog, cat) (dog, litter), (ate, cat) (ate, litter), (cat, litter)
# MAGIC     (the, cat) (the, has) (the, clean) (the, litter), (cat, has) (cat, clean) (cat, litter), (has, clean) (has, litter) (clean, litter)
# MAGIC     
# MAGIC     Unique neighbors:
# MAGIC     the 6
# MAGIC     dog 4
# MAGIC     ate 4
# MAGIC     litter 6
# MAGIC     cat 6
# MAGIC     has 4
# MAGIC     clean 4
# MAGIC  </pre>
# MAGIC     
# MAGIC     
# MAGIC   * the top 10 words with the most "neighbors"
# MAGIC   * the bottom 10 words with least "neighbors"
# MAGIC   * a random sample of 1% of the words' neighbor counts     
# MAGIC   __`NOTE:`__ for the last item, please return only the counts and not the words -- we'll go on to use these in a plotting function that expects a list of integers.
# MAGIC 
# MAGIC 
# MAGIC * __b) short response:__ Use the provided code to plot a histogram of the sampled list from `a`. Comment on the distribution you observe. How will this distribution affect our synonym detection analysis?
# MAGIC 
# MAGIC * __c) code + short response:__ Write a Spark Job to compare word frequencies to number of neighbors.
# MAGIC     * Of the 1000 words with most neighbors, what percent are also in the list of 1000 most frequent words?
# MAGIC     * Of the 1000 words with least neighbors, what percent are also in the list of 1000 least frequent words?   
# MAGIC [__`NOTE:`__ _technically these lists are short enough to compare in memory on your local machine but please design your Spark job as if we were potentially comparing much larger lists._]

# COMMAND ----------

# MAGIC %md ### Q6 Student Answers:
# MAGIC 
# MAGIC > __b)__ Type your answer here!   
# MAGIC 
# MAGIC > __c)__ Type your answer here!

# COMMAND ----------

# part a - spark job
def EDA2(rdd,n):
    top_n, bottom_n, sampled_counts = None, None, None
    ############# YOUR CODE HERE ###############

    
    
    
    
    
    
    
    
    
    
    ############# (END) YOUR CODE ##############
    return top_n, bottom_n, sampled_counts

# COMMAND ----------

# part a - systems test (RUN THIS CELL AS IS)
start = time.time()
most_nbrs, least_nbrs, sample_counts = EDA2(testRDD, 10)
print("Wall time: {} seconds".format(time.time() - start))
# Command took 0.59 seconds -- by kylehamilton@ischool.berkeley.edu at 9/19/2020, 4:53:14 PM on kyle_hw

# COMMAND ----------

# part a - display results (feel free to modify the formatting code if needed)
print(" --- Most Co-Words ---|--- Least Co-Words ----")
for (w1, c1), (w2, c2) in zip(most_nbrs, least_nbrs):
    print(f"{w1:>12} {c1:>8} |{w2:>16} {c2:>4}")

# COMMAND ----------

# MAGIC %md Expected output for testRDD:
# MAGIC <pre>
# MAGIC  --- Most Co-Words ---|--- Least Co-Words ----
# MAGIC          was        9 |     foolishness    4
# MAGIC           of        9 |            best    5
# MAGIC          the        9 |           worst    5
# MAGIC           it        8 |          wisdom    5
# MAGIC          age        7 |             age    7
# MAGIC        times        7 |           times    7
# MAGIC         best        5 |              it    8
# MAGIC        worst        5 |             was    9
# MAGIC       wisdom        5 |              of    9
# MAGIC  foolishness        4 |             the    9
# MAGIC  </pre>

# COMMAND ----------

# part a - single file test (RUN THIS CELL AS IS)
start = time.time()
most_nbrs, least_nbrs, sample_counts = EDA2(f1RDD, 10)
print("Wall time: {} seconds".format(time.time() - start))
# Command took 6.34 seconds -- by kylehamilton@ischool.berkeley.edu at 9/19/2020, 4:53:42 PM on kyle_hw

# COMMAND ----------

# part a - display results (feel free to modify the formatting code if needed)
print(" --- Most Co-Words ---|--- Least Co-Words ----")
for (w1, c1), (w2, c2) in zip(most_nbrs, least_nbrs):
    print(f"{w1:>12} {c1:>8} |{w2:>16} {c2:>4}")

# COMMAND ----------

# MAGIC %md Expected output for f1RDD:
# MAGIC <pre>
# MAGIC  --- Most Co-Words ---|--- Least Co-Words ----
# MAGIC          the    25548 |              vo    1
# MAGIC           of    22496 |      noncleaved    2
# MAGIC          and    16489 |        premiers    2
# MAGIC           to    14249 |        enclaves    2
# MAGIC           in    13891 |   selectiveness    2
# MAGIC            a    13045 |           trill    2
# MAGIC         that     8011 |           pizza    2
# MAGIC           is     7947 |            hoot    2
# MAGIC         with     7552 |     palpitation    2
# MAGIC           by     7400 |            twel    2
# MAGIC </pre>

# COMMAND ----------

# part a - full data (RUN THIS CELL AS IS)
start = time.time()
most_nbrs, least_nbrs, sample_counts = EDA2(dataRDD, 10)
print("Wall time: {} seconds".format(time.time() - start))
# Command took 4.12 minutes -- by kylehamilton@ischool.berkeley.edu at 9/19/2020, 4:54:10 PM on kyle_hw

# COMMAND ----------

# part a - display results (feel free to modify the formatting code if needed)
print(" --- Most Co-Words ---|--- Least Co-Words ----")
for (w1, c1), (w2, c2) in zip(most_nbrs, least_nbrs):
    print(f"{w1:>12} {c1:>8} |{w2:>16} {c2:>4}")

# COMMAND ----------

# MAGIC %md Expected output for dataRDD: 
# MAGIC (bottom words might vary a little due to ties)
# MAGIC <pre>
# MAGIC  --- Most Co-Words ---|--- Least Co-Words ----
# MAGIC          the   164982 |          cococo    1
# MAGIC           of   155708 |            inin    1
# MAGIC          and   132814 |        charuhas    1
# MAGIC           in   110615 |         ooooooo    1
# MAGIC           to    94358 |           iiiii    1
# MAGIC            a    89197 |          iiiiii    1
# MAGIC           by    67266 |             cnj    1
# MAGIC         with    65127 |            choh    1
# MAGIC         that    61174 |             neg    1
# MAGIC           as    60652 |      cococococo    1
# MAGIC </pre>

# COMMAND ----------

# MAGIC %md __`NOTE:`__ _before running the plotting code below, make sure that the variable_ `sample_counts` _points to the list generated in_ `part a`.

# COMMAND ----------

# part b - plot histogram (RUN THIS CELL AS IS - feel free to modify format)

# removing extreme upper tail for a better visual
counts = np.array(sample_counts)[np.array(sample_counts) < 6000]
t = sum(np.array(sample_counts) > 6000)
n = len(counts)
print("NOTE: we'll exclude the %s words with more than 6000 nbrs in this %s count sample." % (t,n))

# set up figure
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (15,5))

# plot regular hist
ax1.hist(counts, bins=50)
ax1.set_title('Freqency of Number of Co-Words', color='0.1')
ax1.set_facecolor('0.9')
ax1.tick_params(axis='both', colors='0.1')
ax1.grid(True)

# plot log scale hist
ax2.hist(counts, bins=50)
ax2.set_title('(log)Freqency of Number of Co-Words', color='0.1')
ax2.set_facecolor('0.9')
ax2.tick_params(axis='both', colors='0.1')
ax2.grid(True)
plt.yscale('log')


# COMMAND ----------

# part c - spark job
def compareRankings(rdd1, rdd2):
    percent_overlap = None
    ############# YOUR CODE HERE ###############

    
    
    
    
    
    ############# (END) YOUR CODE ##############
    return percent_overlap

# COMMAND ----------

# part c - get lists for comparison (RUN THIS CELL AS IS...)
# (... then change 'testRDD' to 'f1RDD'/'dataRDD' when ready)
total, topWords, bottomWords = EDA1(testRDD, 1000)
topNbrs, bottomNbrs, sample_counts = EDA2(testRDD, 1000)
twRDD = sc.parallelize(topWords)
bwRDD = sc.parallelize(bottomWords)
tnRDD = sc.parallelize(topNbrs)
bnRDD = sc.parallelize(bottomNbrs)
top_overlap = compareRankings(tnRDD, twRDD)
bottom_overlap = compareRankings(bnRDD,bwRDD)
print(f"Of the 1000 words with most neighbors, {top_overlap} percent are also in the list of 1000 most frequent words.")
print(f"Of the 1000 words with least neighbors, {bottom_overlap} percent are also in the list of 1000 least frequent words.")

# COMMAND ----------

# MAGIC %md # Question 7: Basis Vocabulary & Stripes
# MAGIC 
# MAGIC Every word that appears in our data is a potential feature for our synonym detection analysis. However as we've discussed, some are likely to be more useful than others. In this question, you'll choose a judicious subset of these words to form our 'basis vocabulary'. Practically speaking, this means that when we build our stripes, we are only going to keep track of when a term co-occurs with one of these basis words. 
# MAGIC 
# MAGIC 
# MAGIC ### Q7 Tasks:
# MAGIC * __a) multiple choice:__ Suppose we were deciding between two different basis vocabularies: the 1000 most frequent words or the 1000 least frequent words. How would this choice impact the quality of the synonyms we are able to detect? How does this choice relate to the ideas of 'overfitting' or 'underfitting' a training set?
# MAGIC <pre>  
# MAGIC   MULTIPLE CHOICE:
# MAGIC    A. 1000 most frequent words would overfit, while 1000 least frequent words would underfit
# MAGIC    B. 1000 most frequent words would underfit, while 1000 least frequent words would overfit
# MAGIC   
# MAGIC   BONUS: Explain your answer 
# MAGIC </pre>
# MAGIC * __b) short response:__ If we had a much larger dataset, computing the full ordered list of words would be extremely expensive. If we need to none-the-less get an estimate of word frequency in order to decide on a basis vocabulary, what alternative strategy could we take?
# MAGIC 
# MAGIC * __c) multiple choice:__ Run the provided spark job that does the following:
# MAGIC   * tokenizes, removes stopwords and computes a word count on the ngram data
# MAGIC   * subsets the top 10,000 words (these are the terms we'll consider as potential synonyms)
# MAGIC   * subsets words 9,000-9,999 (this will be our 1,000 word basis vocabulary)    
# MAGIC   (to put it another way - of the top 10,000 words, the bottom 1,000 form the basis vocabulary)
# MAGIC   * saves the full 10K word list and the 1K basis vocabulary to file for use in `d`.  
# MAGIC <pre>
# MAGIC   What is another way to describe the Basis Vocabulary in machine learning terms?
# MAGIC   A. Stop-words
# MAGIC   B. Features
# MAGIC   C. Postings
# MAGIC   D. 1000-grams
# MAGIC </pre>
# MAGIC 
# MAGIC * __d) code:__ Write a spark job that builds co-occurrence stripes for the top 10K words in the ngram data using the basis vocabulary you developed in `part c`. This job/function, unlike others so far, should return an RDD (which we will then use in q8).

# COMMAND ----------

# MAGIC %md ### Q7 Student Answers:
# MAGIC > __a)__ Type your answer here!   
# MAGIC > __a-bonus)__ Type your answer here!   
# MAGIC > __b)__ Type your answer here!   
# MAGIC > __c)__ Type your answer here!   

# COMMAND ----------

# part c - provided stopwords (RUN THIS CELL AS IS)
STOPWORDS =  ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 
              'ourselves', 'you', 'your', 'yours', 'yourself', 
              'yourselves', 'he', 'him', 'his', 'himself', 'she', 
              'her', 'hers', 'herself', 'it', 'its', 'itself', 
              'they', 'them', 'their', 'theirs', 'themselves', 
              'what', 'which', 'who', 'whom', 'this', 'that', 
              'these', 'those', 'am', 'is', 'are', 'was', 'were', 
              'be', 'been', 'being', 'have', 'has', 'had', 'having', 
              'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 
              'but', 'if', 'or', 'because', 'as', 'until', 'while', 
              'of', 'at', 'by', 'for', 'with', 'about', 'against', 
              'between', 'into', 'through', 'during', 'before', 
              'after', 'above', 'below', 'to', 'from', 'up', 'down', 
              'in', 'out', 'on', 'off', 'over', 'under', 'again', 
              'further', 'then', 'once', 'here', 'there', 'when', 
              'where', 'why', 'how', 'all', 'any', 'both', 'each', 
              'few', 'more', 'most', 'other', 'some', 'such', 'no', 
              'nor', 'not', 'only', 'own', 'same', 'so', 'than', 
              'too', 'very', 'should', 'can', 'now', 'will', 'just', 
              'would', 'could', 'may', 'must', 'one', 'much', "it's",
              "can't", "won't", "don't", "shouldn't", "hasn't"]

# COMMAND ----------

# part c - get the vocabulary and basis (RUN THIS CELL AS IS)
# 
def get_vocab(rdd, n_total, n_basis):
    vocab, basis = None, None
    ############# YOUR CODE HERE ###############
    sw = sc.broadcast(set(STOPWORDS))
    top_10k = rdd.map(lambda line: line.split('\t')[0:2])\
                 .flatMap(lambda x: [(w, int(x[1])) for w in x[0].lower().split() 
                                     if w not in sw.value])\
                 .reduceByKey(lambda x,y: x+y)\
                 .takeOrdered(n_total, key=lambda x: -x[1])
    vocab = [pair[0] for pair in top_10k]
    basis = vocab[n_total - n_basis:]
    ############# (END) YOUR CODE ##############
    return vocab, basis

# COMMAND ----------

# part c - run your job (RUN THIS CELL AS IS)
start = time.time()
VOCAB, BASIS = get_vocab(dataRDD, 10000, 1000)
print("Wall time: {} seconds".format(time.time() - start))
# Command took 1.08 minutes -- by kylehamilton@ischool.berkeley.edu at 9/19/2020, 5:05:32 PM on kyle_hw

# COMMAND ----------

dbutils.fs.put(hw3_path+"vocabulary.txt",str(VOCAB),True)
dbutils.fs.put(hw3_path+"basis.txt",str(BASIS),True)

# COMMAND ----------

# part d - spark job
def buildStripes(rdd, vocab, basis):
    stripesRDD = None
    ############# YOUR CODE HERE ###############

    
    
    
    
    
    
    ############# (END) YOUR CODE ##############
    return stripesRDD

# COMMAND ----------

# part d - run your systems test (RUN THIS CELL AS IS)
VOCAB, BASIS = get_vocab(testRDD, 10, 10)
testStripesRDD = buildStripes(testRDD, VOCAB, BASIS)
start = time.time()
print(testStripesRDD.collect())
print("Wall time: {} seconds".format(time.time() - start))
# Command took 0.44 seconds -- by kylehamilton@ischool.berkeley.edu at 9/19/2020, 5:07:36 PM on kyle_hw
# Expected results
'''
[('worst', {'times'}), ('best', {'times'}), ('foolishness', {'age'}), ('age', {'wisdom', 'foolishness', 'times'}), ('wisdom', {'age'}), ('times', {'age', 'best', 'worst'})]
'''

# COMMAND ----------

# part d - run your single file test (RUN THIS CELL AS IS)
VOCAB, BASIS = get_vocab(f1RDD, 10000, 1000)
f1StripesRDD = buildStripes(f1RDD, VOCAB, BASIS).cache()
start = time.time()
print(f1StripesRDD.top(5))
print("Wall time: {} seconds".format(time.time() - start))
# Command took 3.02 seconds -- by kylehamilton@ischool.berkeley.edu at 9/19/2020, 5:09:34 PM on kyle_hw
# Expected results
'''
[('zippor', {'balak'}), ('zedong', {'mao'}), ('zeal', {'infallibility'}), ('youth', {'mould', 'constrained'}), ('younger', {'careers'})]
'''

# COMMAND ----------

# part d - run the full analysis and take a look at a few stripes (RUN THIS CELL AS IS)
VOCAB = ast.literal_eval(open(hw3_path_open+"vocabulary.txt", "r").read())
BASIS = ast.literal_eval(open(hw3_path_open+"basis.txt", "r").read())
stripesRDD = buildStripes(dataRDD, VOCAB, BASIS).cache()

start = time.time()
for wrd, stripe in stripesRDD.top(3):
    print(wrd)
    print(list(stripe))
    print('-------')
print("Wall time: {} seconds".format(time.time() - start))
# Command took 55.51 seconds -- by kylehamilton@ischool.berkeley.edu at 9/19/2020, 5:11:38 PM on kyle_hw
# Expected results:
'''
zones
['remotest', 'adhesion', 'residential', 'subdivided', 'environments', 'gaza', 'saturation', 'localities', 'uppermost', 'warmer', 'buffer', 'parks']
-------
zone
['tribal', 'narrower', 'fibrous', 'saturation', 'originate', 'auxiliary', 'ie', 'buffer', 'transitional', 'turbulent', 'vomiting', 'americas', 'articular', 'poorly', 'intervening', 'officially', 'accumulate', 'assisting', 'flexor', 'traversed', 'unusually', 'uppermost', 'cartilage', 'inorganic', 'illuminated', 'glowing', 'contamination', 'trigger', 'masculine', 'defines', 'avoidance', 'residential', 'southeastern', 'penis', 'cracks', 'atlas', 'excitation', 'persia', 'diffuse', 'subdivided', 'alaska', 'guides', 'au', 'sandy', 'penetrating', 'parked']
-------
zinc
['ammonium', 'coating', 'pancreas', 'insoluble', "alzheimer's", 'diamond', 'radioactive', 'metallic', 'weighing', 'dysfunction', 'wasting', 'phosphorus', 'transcription', 'dipped', 'hydroxide', 'burns', 'leukemia', 'dietary']
-------
'''

# COMMAND ----------

# part d - save your full stripes to file for ease of retrival later... (OPTIONAL)
dbutils.fs.rm(hw3_path+'stripes',true)
stripesRDD.saveAsTextFile(hw3_path+'stripes')

# COMMAND ----------

dbutils.fs.ls(hw3_path+'stripes')

# COMMAND ----------

# MAGIC %md # Question 8: Synonym Detection
# MAGIC 
# MAGIC We're now ready to perform the main synonym detection analysis. In the tasks below you will compute cosine, jaccard, dice and overlap similarity measurements for each pair of words in our vocabulary and then sort your results to find the most similar pairs of words in this dataset. __`IMPORTANT:`__ When you get to the sorting step please __sort on cosine similarity__ only, so that we can ensure consistent results from student to student. 
# MAGIC 
# MAGIC Remember to test each step of your work with the small files before running your code on the full dataset. This is a computationally intense task: well designed code can be the difference between a 20min job and a 2hr job. __`NOTE:`__ _as you are designing your code you may want to review questions 3 and 4 where we modeled some of the key pieces of this analysis._
# MAGIC 
# MAGIC ### Q8 Tasks:
# MAGIC * __a) short response:__ In question 7 you wrote a function that would create word stripes for each `term` in our vocabulary. These word stripes are essentially an 'embedded representation' of the `term`'s meaning. What is the 'feature space' for this representation? (i.e. what are the features of our 1-hot encoded vectors?). What is the maximum length of a stripe?
# MAGIC 
# MAGIC * __b) short response:__ Remember that we are going to treat these stripes as 'documents' and perform similarity analysis on them. The first step is to emit postings which then get collected to form an 'inverted index.' How many rows will there be in our inverted index? Explain.
# MAGIC 
# MAGIC * __c) short response:__ In the demo from question 2, we were able to compute the cosine similarity directly from the stripes (we did this using their vector form, but could have used the list instead). So why do we need the inverted index?
# MAGIC 
# MAGIC * __d) code:__ Write a spark job that does the following:
# MAGIC   * loops over the stripes from Q7 and emits postings for the `term` _(key:term, value:posting)_   
# MAGIC   * aggregates the postings to create an inverted index _(key:term, value:list of postings)_
# MAGIC   * loops over all pairs of `term`s that appear in the same postings list and emits co-occurrence counts
# MAGIC   * aggregates co-occurrences _(key:word pair, value:count + other payload)_
# MAGIC   * uses the counts (along with the accompanying information) to compute the cosine, jacard, dice and overlap similarity metrics for each pair of words in the vocabulary 
# MAGIC   * retrieve the top 20 and bottom 20 most/least similar pairs of words
# MAGIC   * also return the cached sorted RDD for use in the next question  
# MAGIC   __`NOTE 1`:__ _Don't forget to include the stripe length when you are creating the postings & co-occurrence pairs. A composite key is the way to go here._  
# MAGIC   __`NOTE 2`:__ _Please make sure that your final results are sorted according to cosine similarity otherwise your results may not match the expected result & you will be marked wrong._
# MAGIC   
# MAGIC * __e) code:__ Comment on the quality of the "synonyms" your analysis comes up with. Do you notice anything odd about these pairs of words? Discuss at least one idea for how you might go about improving on the analysis.

# COMMAND ----------

# MAGIC %md ### Q8 Student Answers:
# MAGIC > __a)__ Type your answer here!
# MAGIC 
# MAGIC > __b)__ Type your answer here!
# MAGIC 
# MAGIC > __c)__ Type your answer here!
# MAGIC 
# MAGIC > __e)__ Type your answer here!

# COMMAND ----------

# helper function for pretty printing (RUN THIS CELL AS IS)
def displayOutput(lines):
    template = "{:25}|{:6}, {:7}, {:7}, {:5}"
    print(template.format("Pair", "Cosine", "Jaccard", "Overlap", "Dice"))
    for pair, scores in lines:
        scores = [round(s,4) for s in scores]
        print(template.format(pair, *scores))

# COMMAND ----------

# MAGIC %md __`TIP:`__ Feel free to define helper functions within the main function to help you organize your code. Readability is important! Eg:
# MAGIC ```
# MAGIC def similarityAnlysis(stripesRDD):
# MAGIC     """main docstring"""
# MAGIC     
# MAGIC     simScoresRDD, top_n, bottom_n = None, None, None
# MAGIC     
# MAGIC     ############ YOUR CODE HERE ###########
# MAGIC     def helper1():
# MAGIC         """helper docstring"""
# MAGIC         return x
# MAGIC         
# MAGIC     def helper2():
# MAGIC         """helper docstring"""
# MAGIC         return x
# MAGIC         
# MAGIC     # main spark job starts here
# MAGIC     
# MAGIC         ...etc
# MAGIC     ############ (END) YOUR CODE ###########
# MAGIC     return simScoresRDD, top_n, bottom_n
# MAGIC ```

# COMMAND ----------

# part d - write your spark job in the space provided
def similarityAnalysis(stripesRDD, n):
    """
    This function defines a Spark DAG to compute cosine, jaccard, 
    overlap and dice scores for each pair of words in the stripes
    provided. 
    
    Output: an RDD, a list of top n, a list of bottom n
    """
    simScoresRDD, top_n, bottom_n = None, None, None
    
    ############### YOUR CODE HERE ################
   






















    ############### (END) YOUR CODE ##############
    return result, top_n, bottom_n

# COMMAND ----------

# part d - run the system test (RUN THIS CELL AS IS... use display cell below to see results)
start = time.time()
testResult, top_n, bottom_n = similarityAnalysis(testStripesRDD, 10)
print("Wall time: {} seconds".format(time.time() - start))
# Command took 0.40 seconds -- by kylehamilton@ischool.berkeley.edu at 9/19/2020, 5:15:03 PM on kyle_hw

# COMMAND ----------

# part d - run the system test (RUN THIS CELL AS IS... use display cell below to see results)
start = time.time()
f1Result, top_n, bottom_n = similarityAnalysis(f1StripesRDD, 10)
print("Wall time: {} seconds".format(time.time() - start))
# Command took 1.11 seconds -- by kylehamilton@ischool.berkeley.edu at 9/19/2020, 5:15:34 PM on kyle_hw

# COMMAND ----------

displayOutput(top_n)

# COMMAND ----------

displayOutput(bottom_n)

# COMMAND ----------

# part d - run the system test (RUN THIS CELL AS IS... use display cell below to see results)
start = time.time()
result, top_n, bottom_n = similarityAnalysis(stripesRDD, 20)
print("Wall time: {} seconds".format(time.time() - start))
# Command took 4.66 minutes -- by kylehamilton@ischool.berkeley.edu at 9/19/2020, 5:16:00 PM on kyle_hw

# COMMAND ----------

displayOutput(top_n)

# COMMAND ----------

displayOutput(bottom_n)

# COMMAND ----------

# MAGIC %md __Expected output f1RDD:__  
# MAGIC <table>
# MAGIC <th>MOST SIMILAR:</th>
# MAGIC <th>LEAST SIMILAR:</th>
# MAGIC <tr><td><pre>
# MAGIC Pair                     |Cosine, Jaccard, Overlap, Dice 
# MAGIC commentary - lady        |   1.0,     1.0,     1.0,   1.0
# MAGIC commentary - toes        |   1.0,     1.0,     1.0,   1.0
# MAGIC commentary - reply       |   1.0,     1.0,     1.0,   1.0
# MAGIC curious - tone           |   1.0,     1.0,     1.0,   1.0
# MAGIC curious - lady           |   1.0,     1.0,     1.0,   1.0
# MAGIC curious - owe            |   1.0,     1.0,     1.0,   1.0
# MAGIC lady - tone              |   1.0,     1.0,     1.0,   1.0
# MAGIC reply - tone             |   1.0,     1.0,     1.0,   1.0
# MAGIC lady - toes              |   1.0,     1.0,     1.0,   1.0
# MAGIC lady - reply             |   1.0,     1.0,     1.0,   1.0
# MAGIC </pre></td>
# MAGIC <td><pre>
# MAGIC 
# MAGIC Pair                     |Cosine, Jaccard, Overlap, Dice 
# MAGIC part - time              |0.0294,  0.0149,  0.0303, 0.0294
# MAGIC time - upon              |0.0314,  0.0159,  0.0345, 0.0312
# MAGIC time - two               |0.0314,  0.0159,  0.0345, 0.0312
# MAGIC made - time              |0.0325,  0.0164,   0.037, 0.0323
# MAGIC first - time             |0.0338,  0.0169,    0.04, 0.0333
# MAGIC new - time               |0.0352,  0.0175,  0.0435, 0.0345
# MAGIC part - us                |0.0355,  0.0179,  0.0417, 0.0351
# MAGIC little - part            |0.0355,  0.0179,  0.0417, 0.0351
# MAGIC made - two               |0.0357,  0.0182,   0.037, 0.0357
# MAGIC made - upon              |0.0357,  0.0182,   0.037, 0.0357
# MAGIC </pre></td></tr>
# MAGIC </table>
# MAGIC 
# MAGIC __Expected output dataRDD:__  
# MAGIC <table>
# MAGIC <th>Most Similar</th>
# MAGIC <th>Least Similar</th>
# MAGIC <tr><td><pre>
# MAGIC Pair                     |Cosine, Jaccard, Overlap, Dice 
# MAGIC first - time             |  0.89,  0.8012,  0.9149, 0.8897
# MAGIC time - well              |0.8895,   0.801,   0.892, 0.8895
# MAGIC great - time             | 0.875,  0.7757,   0.925, 0.8737
# MAGIC part - well              | 0.874,  0.7755,  0.9018, 0.8735
# MAGIC first - well             |0.8717,  0.7722,  0.8936, 0.8715
# MAGIC part - time              |0.8715,  0.7715,  0.9018, 0.871
# MAGIC time - upon              |0.8668,   0.763,  0.9152, 0.8656
# MAGIC made - time              | 0.866,  0.7619,  0.9109, 0.8649
# MAGIC made - well              |0.8601,  0.7531,  0.9022, 0.8592
# MAGIC time - way               |0.8587,  0.7487,  0.9259, 0.8563
# MAGIC great - well             |0.8526,  0.7412,  0.8988, 0.8514
# MAGIC time - two               |0.8517,  0.7389,  0.9094, 0.8498
# MAGIC first - great            |0.8497,  0.7381,  0.8738, 0.8493
# MAGIC first - part             |0.8471,  0.7348,  0.8527, 0.8471
# MAGIC great - upon             |0.8464,  0.7338,  0.8475, 0.8464
# MAGIC upon - well              |0.8444,   0.729,   0.889, 0.8433
# MAGIC new - time               |0.8426,   0.724,  0.9133, 0.8399
# MAGIC first - two              |0.8411,  0.7249,  0.8737, 0.8405
# MAGIC way - well               |0.8357,  0.7146,  0.8986, 0.8335
# MAGIC time - us                |0.8357,  0.7105,  0.9318, 0.8308
# MAGIC 
# MAGIC </pre></td>
# MAGIC <td><pre>
# MAGIC Pair                     |Cosine, Jaccard, Overlap, Dice 
# MAGIC region - write           |0.0067,  0.0032,  0.0085, 0.0065
# MAGIC relation - snow          |0.0067,  0.0026,  0.0141, 0.0052
# MAGIC cardiac - took           |0.0074,  0.0023,  0.0217, 0.0045
# MAGIC ever - tumor             |0.0076,   0.002,  0.0263, 0.004
# MAGIC came - tumor             |0.0076,   0.002,  0.0263, 0.004
# MAGIC let - therapy            |0.0076,   0.003,  0.0161, 0.0059
# MAGIC related - stay           |0.0078,  0.0036,  0.0116, 0.0072
# MAGIC factors - hear           |0.0078,  0.0039,  0.0094, 0.0077
# MAGIC implications - round     |0.0078,  0.0033,  0.0145, 0.0066
# MAGIC came - proteins          |0.0079,   0.002,  0.0286, 0.0041
# MAGIC population - window      |0.0079,  0.0039,    0.01, 0.0077
# MAGIC love - proportional      | 0.008,  0.0029,  0.0185, 0.0058
# MAGIC got - multiple           | 0.008,  0.0034,  0.0149, 0.0067
# MAGIC changes - fort           |0.0081,  0.0032,  0.0161, 0.0065
# MAGIC layer - wife             |0.0081,  0.0038,  0.0119, 0.0075
# MAGIC five - sympathy          |0.0081,  0.0034,  0.0149, 0.0068
# MAGIC arrival - essential      |0.0081,   0.004,  0.0093, 0.008
# MAGIC desert - function        |0.0081,  0.0031,  0.0175, 0.0062
# MAGIC fundamental - stood      |0.0081,  0.0038,  0.0115, 0.0077
# MAGIC patients - plain         |0.0081,   0.004,  0.0103, 0.0079
# MAGIC </pre></td></tr>
# MAGIC </table>

# COMMAND ----------

# MAGIC %md ### Congratulations, you have completed HW3! Please refer to the readme for submission instructions.
# MAGIC 
# MAGIC If you would like to provide feedback regarding this homework, please use the survey at: https://docs.google.com/forms/d/e/1FAIpQLSce9feiQeSkdP43A0ZYui1tMGIBfLfzb0rmgToQeZD9bXXX8Q/viewform

# COMMAND ----------


