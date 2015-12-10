#Apache Spark X-Means Java Implementation

sparkXMeans is a Java implementation of X-means clustering algorithm on Apache spark platform. The algorithm implements the concepts described in:

Pelleg, Dan, and Andrew W. Moore. "X-means: Extending K-means with Efficient Estimation of the Number of Clusters." In ICML, pp. 727-734. 2000.

https://www.cs.cmu.edu/~dpelleg/download/xmeans.pdf

An additional function has been added to calculate BIC values for a range of k values and select k with highest. This implementation is, of course, slower than the original one.

