from __future__ import print_function
import ast
import sys
from pyspark.accumulators import AccumulatorParam
from pyspark.sql import SparkSession

# Spark only implements Accumulator parameter for numeric types.
# This class extends Accumulator support to the string type.
class StringAccumulatorParam(AccumulatorParam):
    def zero(self, value):
        return value
    def addInPlace(self, val1, val2):
        return val1 + val2



###################################
# INITIALIZE GRAPH
###################################

def parseData(line):
    line = line.strip()
    key, value = line.split("\t")
    key = str(key)
  
    if key == startNode.value:
        return (key, ("Q",ast.literal_eval(value),0,key))
    else:
        return (key, ("U",ast.literal_eval(value),float("inf"),""))
    
    
    
###################################
# MAPPER
###################################   

def expandFrontier(row):
  key = row[0]
  status = row[1][0]
  neighbors = row[1][1]
  distance = row[1][2]
  path = row[1][3]
  
  if status == "Q":
    
    # put neighbors in Q mode and update path length by incrementing path length of N
    for neighbor in neighbors:
      yield neighbor, ("Q", {}, distance + int(neighbors[neighbor]), str(path)+" -> "+str(neighbor))
      
    # Update status of current node to Visited
    status = "V"
      
  yield key, (status, neighbors, distance, path)

  
###################################
# REDUCER
###################################

def restoreGraph(a,b):
    
    # It's important that the node in status Q comes first.
    a,b = sorted([a,b]) 
    
    _status, _neighbors, _distance, _path = a # <- Q state (if there is a Q state)
    status, neighbors, distance, path = b # <- V or U state

    if distance > _distance: # if the new path we discovered is shorter than the distance in a visited node, reset the visited node to Q state
        status = "Q" # <- the magic for weighted graphs
        distance = _distance
        path = _path            

    return (status, neighbors, distance, path)  

  
###################################
# ACCUMULATORS
###################################  
  
def terminate(row):
  if row[1][0] == "V" and row[0] == targetNode.value:  
    targetAccum.add(1)
    pathAccum.add(str(row[1][3])+" distance: "+str(row[1][2]))
  if row[1][0] == "Q":
    statusAccum.add(1)

    
    
if __name__ == "__main__":
  
  if len(sys.argv) != 5:
    print("Usage: SSSP <file> <startNode> <targetNode> <isWeighted: 0|1>", file=sys.stderr)
    sys.exit(-1)

    
  
  app_name = "graphs-intro"
  master = "local[*]"
  
  spark = SparkSession \
          .builder \
          .appName(app_name) \
          .master(master) \
          .getOrCreate()
  
  sc = spark.sparkContext
  
  # remember to broadcast global variables:
  dataFile = sc.textFile(sys.argv[1])
  startNode = sc.broadcast(sys.argv[2])
  targetNode = sc.broadcast(sys.argv[3])
  weighted = sys.argv[4]
  
  rdd = dataFile.map(parseData).cache()

  notconverged = True
  iteration = 0
  while notconverged:
    iteration = iteration + 1
    targetAccum = sc.accumulator(0)
    statusAccum = sc.accumulator(0)
    pathAccum = sc.accumulator("", StringAccumulatorParam())

    rdd = rdd.flatMap(expandFrontier).reduceByKey(restoreGraph)

    rdd.foreach(terminate)
  
    if weighted == "1":
      if statusAccum.value == 0: # no more nodes in Q status
        notconverged = False
    else:
      if targetAccum.value == 1: # reached target node
        notconverged = False
        
    print("-"*50)  
    print ("After Iteration "+str(iteration))
    print("Node id, (Status, {out_nodes},distance,path)")
    
    for i in rdd.collect():
      print(i)
      
    print("Num nodes in Q status: ",statusAccum.value)
    #print("Target node in V status: ",targetAccum.value)  # we only care about this in unweighted graphs, where reaching target node terminates the algorithim
    print("-"*50)    
    

  print("Num nodes in Q status: ",statusAccum.value)
  #print("Target node in V status: ",targetAccum.value)
  print("Iterations: ", iteration)
  print("Path: ",pathAccum.value)
  print("="*20)

  spark.stop()