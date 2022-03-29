#!/usr/bin/env python
"""
This script reads lines from STDIN and returns a list of
all words an the count of how many times they occurred.

INPUT:
    a text file
OUTPUT FORMAT:
    word \t count
USAGE:
    python wordCount.py < yourTextFile.txt

Instructions:
    Fill in the missing code below so that the script
    prints tab separated word counts to Standard Output.
    NOTE: we have performed the tokenizing for you, please
    don't modify the provided code or you may fail unittests.
"""

# imports
import sys
import re
from collections import defaultdict

counts = defaultdict(int)

# stream over lines from Standard Input
for line in sys.stdin:

    # tokenize
    line = line.strip()
    words = re.findall(r'[a-z]+', line.lower())

############ YOUR CODE HERE #########

    # for each distinct word in 'words', incremen count by 1.  
    for word in words:
        counts[word] = counts[word] + 1

# print tab-seperated list of words with respective counts
[print(word[0], "\t", word[1]) for word in counts.items()]






############ (END) YOUR CODE #########
