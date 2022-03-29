#!/usr/bin/env python
"""
Reducer aggregates word counts by class and emits frequencies.

INPUT:
    partitionKey \t word \t class0_partialCount,class1_partialCount 
OUTPUT:
    ID \t word \t class0_count,class1count,P(word|class0),P(word|class1)
    
Instructions:
    Again, you are free to design a solution however you see 
    fit as long as your final model meets our required format
    for the inference job we designed in Question 8. Please
    comment your code clearly and concisely.
    
    A few reminders: 
    1) Don't forget to emit Class Priors (with the right key).
    2) In python2: 3/4 = 0 and 3/float(4) = 0.75
"""
##################### YOUR CODE HERE ####################


import re                                                   
import sys                                                  
import numpy as np      

from operator import itemgetter
import os

# initialize variables
cur_word = None
class0_count, class0_totalWords, class0_totalDocs = 0, 0, 0
class1_count, class1_totalWords, class1_totalDocs = 0, 0, 0
totalDocs, totalWords = 0, 0

# read input key-value pairs from standard input
for line in sys.stdin:
    
    line.rstrip('\n')
    pKey, word, value = line.split('\t',2)
    value_tuple = tuple(value.split (","))
    valueTemp0 = int(value_tuple[0])
    valueTemp1 = int(value_tuple[1])

    # tally counts from current key
    if word == cur_word:     
        class0_count += valueTemp0
        class1_count += valueTemp1
    else:
        # total words / class type word total
        if cur_word == "!totalWordCount":
            totalWords = class0_count + class1_count
            class0_totalWords = class0_count
            class1_totalWords = class1_count
        
        # total docs / class type doc total
        if cur_word == "!totalDocClassCount":
            totalDocs = class0_count + class1_count
            class0_totalDocs = class0_count
            class1_totalDocs = class1_count
            
        if (cur_word and cur_word != "!totalWordCount" and cur_word != "!totalDocClassCount"):
            print(f'{cur_word}\t{float(class0_count)}, {float(class1_count)}, \
                 {float(class0_count/class0_totalWords)}, {float(class1_count/class1_totalWords)}')
        cur_word, class0_count, class1_count = word, int(value_tuple[0]), int(value_tuple[1])

print(f'{cur_word}\t{float(class0_count)}, {float(class1_count)}, \
     {float(class0_count/class0_totalWords)}, {float(class1_count/class1_totalWords)}')

print(f'ClassPriors\t{float(class0_totalDocs)}, {float(class1_totalDocs)}, \
     {float(class0_totalDocs/totalDocs)}, {float(class1_totalDocs/totalDocs)}')

##################### (END) CODE HERE ####################