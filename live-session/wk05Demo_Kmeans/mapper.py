#!/usr/bin/env python
import sys, math

CLUSTERS_FILENAME = 'centroids.txt'

centroid_points = [list(map(float, s.split('\n')[0].split(','))) 
                   for s 
                   in open(CLUSTERS_FILENAME, 'r').readlines()]

for line in sys.stdin:
    data = [float(x) 
            for x 
            in line.strip().split(',')]

    minDistance, index = 0, -1
    for i in range(len(centroid_points)):
        centroid = centroid_points[i]
        distance = sum([(centroid[ix]-data[ix])**2 
                    for ix 
                    in range(len(data))])**2
        if minDistance:
            if distance < minDistance:                    
                minDistance, index = distance, i
        else:
            minDistance, index = distance, i
    print(f"{index}\t{1}\t{line.strip()}")