#!/usr/bin/env python

import os
import sys                                                  
import numpy as np  

#################### YOUR CODE HERE ###################


# initialize trackers
cur_word = None
totalTerms, totalWords = 0, 0
class0_totalWords, class1_totalWords = 0, 0
count0, count1, count2, count3 = 0, 0, 0, 0
init_count0, init_count1, init_count2, init_count3 = 0, 0, 0, 0

# read input key-value pairs from standard input
for line in sys.stdin:
    
    line.rstrip('\n')
    word, value = line.split('\t',1)
    value_tuple = tuple(value.split (","))

    value0, value1, value2, value3 = \
        float(value_tuple[0]), float(value_tuple[1]), \
        float(value_tuple[2]), float(value_tuple[3])
    
    # tally counts from current key
    if word == cur_word:     
        count0 += value0
        count1 += value1
        count2 += value2
        count3 += value3
    else:
        # get totals
        if cur_word == "!TotalTerms":
            totalTerms = count0
            class0_totalWords = count2
            class1_totalWords = count3

        if cur_word == "ClassPriors":
            count0, count1, count2, count3 = init_count0, init_count1, \
                init_count2, init_count3
            totalWords = count0 + count1

            print(f'{cur_word}\t{float(count0)},{float(count1)},\
                 {float(count0/totalWords)},{float(count1/totalWords)}')        
        if (cur_word and cur_word != "!TotalTerms" and cur_word != "ClassPriors"):

            print(f'{cur_word}\t{float(count0)}, {float(count1)}, \
                 {float((count0+1)/(class0_totalWords+totalTerms))}, {float((count1+1)/(class1_totalWords+totalTerms))}') 
        cur_word, count0, count1, count2, count3 = \
           word, float(value_tuple[0]), float(value_tuple[1]), float(value_tuple[2]), float(value_tuple[3])
    init_count0, init_count1, init_count2, init_count3 = value0, value1, value2, value3

print(f'{cur_word}\t{float(count0)}, {float(count1)}, \
     {float((count0+1)/(class0_totalWords+totalTerms))}, {float((count1+1)/(class1_totalWords+totalTerms))}') 

#################### (END) YOUR CODE ###################