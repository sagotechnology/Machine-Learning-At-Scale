#!/usr/bin/env python
import sys, re, math
from operator import add

oldKey = None
sumOfCoords = None
countOfCoords = 0

emit = lambda i, s, c: '\t'.join([str(i), str(c), ','.join([str(x) for x in s])])

for line in sys.stdin:
    line = line.strip().split('\t')
    index, count, coords  = int(line[0]), int(line[1]), [float(x) 
                                                         for x 
                                                         in line[2].split(',')]
    if oldKey is not None and oldKey != index:
        print(emit(oldKey, sumOfCoords, countOfCoords))
        sumOfCoords = None
        countOfCoords = 0
    
    oldKey = index
    sumOfCoords = map(add, sumOfCoords, coords) if sumOfCoords else coords
    countOfCoords += int(count)

if oldKey != None:
    print(emit(oldKey, sumOfCoords, countOfCoords))