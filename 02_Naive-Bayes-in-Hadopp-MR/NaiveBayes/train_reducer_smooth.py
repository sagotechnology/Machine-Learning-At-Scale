#!/usr/bin/env python

import os
import sys                                                  
import numpy as np  

#################### YOUR CODE HERE ###################

# initialize variables
cur_word = None
class0_count, class0_totalDocs, class0_totalTerms, class0_totalWords = 0, 0, 0, 0
class1_count, class1_totalDocs, class1_totalTerms, class1_totalWords = 0, 0, 0, 0
totalTerms, totalWords, valueTemp0, valueTemp1 = 1, 0, None, None

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
            totalTerms += 1
            class0_totalTerms += class0_count
            class1_totalTerms += class1_count
        cur_word, class0_count, class1_count = word, int(value_tuple[0]), int(value_tuple[1])

class0_totalTerms += class0_count
class1_totalTerms += class1_count

print(f'{cur_word}\t{float(class0_count)}, {float(class1_count)}, \
     {float(class0_count/class0_totalWords)}, {float(class1_count/class1_totalWords)}')
     
print(f'ClassPriors\t{float(class0_totalDocs)}, {float(class1_totalDocs)}, \
     {float(class0_totalDocs/totalDocs)}, {float(class1_totalDocs/totalDocs)}')
     
print(f'!TotalTerms\t{float(totalTerms)}, {float(totalTerms)}, \
     {float(class0_totalTerms)}, {float(class1_totalTerms)}')

#################### (END) YOUR CODE ###################