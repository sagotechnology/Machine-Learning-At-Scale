#!/usr/bin/env python
"""
Mapper reads in text documents and emits word counts by class.
INPUT:                                                    
    DocID \t true_class \t subject \t body                
OUTPUT:                                                   
    partitionKey \t word \t class0_partialCount,class1_partialCount       
    

Instructions:
    You know what this script should do, go for it!
    (As a favor to the graders, please comment your code clearly!)
    
    A few reminders:
    1) To make sure your results match ours please be sure
       to use the same tokenizing that we have provided in
       all the other jobs:
         words = re.findall(r'[a-z]+', text-to-tokenize.lower())
         
    2) Don't forget to handle the various "totals" that you need
       for your conditional probabilities and class priors.
       
Partitioning:
    In order to send the totals to each reducer, we need to implement
    a custom partitioning strategy.
    
    We will generate a list of keys based on the number of reduce tasks 
    that we read in from the environment configuration of our job.
    
    We'll prepend the partition key by hashing the word and selecting the
    appropriate key from our list. This will end up partitioning our data
    as if we'd used the word as the partition key - that's how it worked
    for the single reducer implementation. This is not necessarily "good",
    as our data could be very skewed. However, in practice, for this
    exercise it works well. The next step would be to generate a file of
    partition split points based on the distribution as we've seen in 
    previous exercises.
    
    Now that we have a list of partition keys, we can send the totals to 
    each reducer by prepending each of the keys to each total.
       
"""

import re                                                   
import sys                                                  
import numpy as np      

from operator import itemgetter
import os



#################### YOUR CODE HERE ###################
def getPartitionKey(word):
    "Helper function to assign partition key alphabetically."
    if word[0] < 'g':
        return 'A'
    elif word[0] < 'n':
        return 'B'
    elif word[0] <'t':
        return 'C'
    else:                                          
        return 'D' 

# initialize variables
current_word = None
class0_partialCount, class1_partialCount = 0, 0
class0_totalWords, class1_totalWords = 0, 0
class0_docCount, class1_docCount = 0, 0

# read from standard input
for line in sys.stdin:
    
    # parse input and tokenize
    docID, _class, subject, body = line.lower().split('\t')
    words = re.findall(r'[a-z]+', subject + ' ' + body)
    if int(_class) == 0:
        class0_docCount += 1
    else:
        class1_docCount += 1
    # emit words and count 
    for word in words:
        class0_partialCount, class1_partialCount = 0, 0
        partitionKey = getPartitionKey(word)
        
        if int(_class) == 0:
            class0_partialCount = 1
            class0_totalWords += 1

        else:
            class1_partialCount = 1
            class1_totalWords += 1

        print(f"{partitionKey}\t{word}\t{class0_partialCount},{class1_partialCount}")


# emit total count to each partition
for pkey in ['A','B','C', 'D']: 
        print(f'{pkey}\t!totalWordCount\t{class0_totalWords},{class1_totalWords}')     
        print(f'{pkey}\t!totalDocClassCount\t{class0_docCount},{class1_docCount}')


#################### (END) YOUR CODE ###################