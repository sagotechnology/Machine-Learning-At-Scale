#!/usr/bin/env python
"""
Reducer takes words with their class and partial counts and computes totals.
INPUT:
    word \t class \t partialCount 
OUTPUT:
    word \t class \t totalCount  
"""
import re
import sys

# initialize trackers
current_word = None
spam_count, ham_count = 0,0

# read from standard input
for line in sys.stdin:
    # parse input
    word, is_spam, count = line.split('\t')
    
############ YOUR CODE HERE #########
    # tally counts from current word by is_spam value
    if word == current_word:
        if int(is_spam) == 0: 
            ham_count += int(count)
        else:
            spam_count += int(count)
    else: 
        # emit ham word count
        if ham_count > 0:
            print(f'{current_word}\t{0}\t{ham_count}')
            
        # emit spam word count
        if spam_count > 0:
            print(f'{current_word}\t{1}\t{spam_count}')
        
        # set/reset spam/ham count for new word    
        if int(is_spam) == 0:             
            current_word, ham_count, spam_count  = word, 1, 0
        else:
            current_word, ham_count, spam_count  = word, 0, 1      
        
# emit last record 
print(f'{current_word}\t{0}\t{ham_count}')
############ (END) YOUR CODE #########