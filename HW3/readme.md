# HW 3 - Synonym Detection In Spark

In the last homework assignment you performed Naive Bayes to classify documents as 'ham' or 'spam.' In doing so, we relied on the implicit assumption that the list of words in a document can tell us something about the nature of that document's content. We'll rely on a similar intuition this week: the idea that, if we analyze a large enough corpus of text, the list of words that appear in small window before or after a vocabulary term can tell us something about that term's meaning. This is similar to the intuition behind the word2vec algorithm.

This will be your first assignment working in Spark. You'll perform Synonym Detection by repurposing an algorithm commonly used in Natural Language Processing to perform document similarity analysis. In doing so you'll also become familiar with important datatypes for efficiently processing sparse vectors and a number of set similarity metrics (e.g. Cosine, Jaccard, Dice). By the end of this homework you should be able to:

... define the terms one-hot encoding, co-occurrance matrix, stripe, inverted index, postings, and basis vocabulary in the context of both synonym detection and document similarity analysis.  
... explain the reasoning behind using a word stripe to compare word meanings.  
... identify what makes set-similarity calculations computationally challenging.  
... implement stateless algorithms in Spark to build stripes, inverted index and compute similarity metrics.  
... identify when it makes sense to take a stripe approach and when to use pairs.  
... apply appropriate metrics to assess the performance of your synonym detection algorithm.  
