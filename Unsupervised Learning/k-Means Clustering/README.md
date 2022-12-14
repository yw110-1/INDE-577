# $k$-Means Clustering
In this sub-repository, I will introduce the k-Means Clustering. K-means is considered as one of the most used clustering algorithms due to its simplicity.

## Algorithm
Kmeans algorithm is an iterative algorithm that tries to partition the dataset into Kpre-defined distinct non-overlapping subgroups/clusters where each data point belongs to only one group. It tries to make the intra-cluster data points as similar as possible while also keeping the clusters as far as possible.
- The way k-means algorithm works is as follows:
  - Specify number of clusters K.
  - Initialize centroids by first shuffling the dataset and then randomly selecting K data points for the centroids without replacement.
  - Keep iterating until there is no change to the centroids. i.e assignment of data points to clusters isn’t changing.
  - Compute the sum of the squared distance between data points and all centroids.
  - Assign each data point to the closest cluster (centroid).
  - Compute the centroids for the clusters by taking the average of the all data points that belong to each cluster.

## Dataset
The dataset I am using is [make_classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html) dataset.

## Reference
1. Dr. Davila's Github, https://github.com/RandyRDavila/Data_Science_and_Machine_Learning_Spring_2022
2. Dabbura, Imad, K-means Clustering: Algorithm, Applications, Evaluation Methods, and Drawbacks, https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a
