# wk3 Live Session Demo: K-Means and Hadoop Limitations

Over the last two weeks you've learned to design MapReduce jobs in Hadoop Streaming. We paid particular attention to the concepts of partitioning and local aggregation as examples of how a programmer can make smart design choices whem implementing parallelized versions of machine learning algorithms. However Hadoop Streaming also has some significant limitations when it comes to offering the programmer design choices. In particular, many Machine Learning tasks require multiple map and reduce phases which can be onerous to implement within the Hadoop Streaming context. Today we'll discuss K-Means as an example of an algorithm that is easy to parallelize but frustrating to implement with Hadoop Streaming due to its iterative nature. This discussion should motivate our transition to Spark next week.


