# Machine Learning at Scale  

## Homework 1 - Intro to the Map Reduce Paradigm
This first homework assignment introduces one of the core strategies in distributed processing: divide and conquer. We'll use the simplest of tasks, word counting, to illustrate the difference between a scalable and non-scalable algorithm. You will be working with the text of Alice in Wonderland to put these ideas into practice using Python and Bash scripting. By the end of this week you should be able to:

... describe the Bias-Variance tradeoff as it applies to Machine Learning.  
... explain why we consider word counting to be an "Embarrassingly Parallel" task.  
... estimate the runtime of embarrassingly parallel tasks using "back of the envelope" calculations.  
... implement a Map Reduce algorithm using the Command Line.  
... set-up a Docker container and know why we use them for this course.  
You will also become familiar (if you aren't already) with defaultdict, re and time in Python, linux piping and sorting, and Jupyter magic commands %%writefile and %%timeit.  

## HW 2 - Naive Bayes in Hadoop MR
In this homework we'll use Hadoop MapReduce to implement your first parallelized machine learning algorithm: Naive Bayes. As you develop your implementation you'll test it on a small dataset that matches the 'Chinese Example' in the Manning, Raghavan and Shutze reading for Week 2. For the main task in this assignment you'll be working with a small subset of the Enron Spam/Ham Corpus. By the end of this assignment you should be able to:

... describe the Naive Bayes algorithm including both training and inference.  
... perform EDA on a corpus using Hadoop MR.  
... implement parallelized Naive Bayes.  
... constrast partial, unordered and total order sort and their implementations in Hadoop Streaming.  
... explain how smoothing affects the bias and variance of a Multinomial Naive Bayes model.  
As always, your work will be graded both on the correctness of your output and on the clarity and design of your code.  

## HW 3 - Synonym Detection In Spark
In the last homework assignment you performed Naive Bayes to classify documents as 'ham' or 'spam.' In doing so, we relied on the implicit assumption that the list of words in a document can tell us something about the nature of that document's content. We'll rely on a similar intuition this week: the idea that, if we analyze a large enough corpus of text, the list of words that appear in small window before or after a vocabulary term can tell us something about that term's meaning. This is similar to the intuition behind the word2vec algorithm.

This will be your first assignment working in Spark. You'll perform Synonym Detection by repurposing an algorithm commonly used in Natural Language Processing to perform document similarity analysis. In doing so you'll also become familiar with important datatypes for efficiently processing sparse vectors and a number of set similarity metrics (e.g. Cosine, Jaccard, Dice). By the end of this homework you should be able to:

... define the terms one-hot encoding, co-occurrance matrix, stripe, inverted index, postings, and basis vocabulary in the context of both synonym detection and document similarity analysis.  
... explain the reasoning behind using a word stripe to compare word meanings.  
... identify what makes set-similarity calculations computationally challenging.  
... implement stateless algorithms in Spark to build stripes, inverted index and compute similarity metrics.  
... identify when it makes sense to take a stripe approach and when to use pairs.  
... apply appropriate metrics to assess the performance of your synonym detection algorithm.  

## HW 4 - Supervised Learning at Scale.

In the first three homeworks you became familiar with the Map-Reduce programming paradigm as manifested in the Hadoop Streaming and Spark frameworks. We explored how different data structures and design patterns can help us manage the computational complexity of an algorithm. As part of this process you implemented both a supervised learning alogorithm (Naive Bayes) and an unsupervised learning algorithm (synonym detection via cosine similarity). In both of these tasks parallelization helped us manage calculations involving a large number of features. However a large feature space isn't the only situation that might prompt us to want to parallelize a machine learning algorithm. In the final two assignments we'll look at cases where the iterative nature of an algorithm is the main driver of its computational complexity (and the reason we might want to parallelize it).

In this week's assignment we'll perform 3 kinds of linear regression: OLS, Ridge and Lasso. As in previous assignments you will implement the core calculations using Spark RDDs... though we've provided more of a code base than before since the focus of the latter half of the course is more on general machine learning concepts. By the end of this homework you should be able to:

... define the loss functions for OLS, Ridge and Lasso regression.  
... calculate the gradient for each of these loss functions.  
... identify which parts of the gradient descent algorithm can be parallelized.  
... implement parallelized gradient descent with cross-validation and regularization.  
... compare/contrast how L1 and L2 regularization impact model parameters & performance.  
