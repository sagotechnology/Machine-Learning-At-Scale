#!/usr/bin/env python
"""
This script reads word counts from STDIN and aggregates
the counts for any duplicated words.

INPUT & OUTPUT FORMAT:
    word \t count
USAGE (standalone):
    python aggregateCounts_v2.py < yourCountsFile.txt

Instructions:
    For Q7 - Your solution should not use a dictionary or store anything   
             other than a single total count - just print them as soon as  
             you've added them. HINT: you've modified the framework script 
             to ensure that the input is alphabetized; how can you 
             use that to your advantage?
"""

# imports
import sys


################# YOUR CODE HERE #################

# intialize word and count
current_word = ""
current_count = 0


# loop through STDIN to get word and count
for line in sys.stdin:
    word, count = line.split()

    # initialize word
    if current_word == "":
        current_word = word
    
    if current_word == word:
        current_count += int(count)
    else:
        print(f"{current_word}\t{current_count}")
        current_word = word
        current_count = int(count)


################ (END) YOUR CODE #################
