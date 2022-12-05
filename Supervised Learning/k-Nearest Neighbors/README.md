# $k$-Nearest Neighbors
In this sub-repository, I will introduce k-Nearest Neighbors Algorithm. The  k-nearest neighbors algorithm, or KNN, is a nonparametric algorithm that assumes that similar data exist in close proximity. In other words, similar things are near to each other.

## Algorithm
KNN captures the idea of similarity (sometimes called distance, proximity, or closeness) with some mathematics we might have learned in our childhood— calculating the distance between points on a graph.

KNN algorithm includes:
1. Load the data.
2. Initialize K to your chosen number of neighbors.
3. For each example in the data:
  - Calculate the distance between the query example and the current example from the data.
  - Add the distance and the index of the example to an ordered collection.
4. Sort the ordered collection of distances and indices from smallest to largest (in ascending order) by the distances.
5. Pick the first K entries from the sorted collection.
6. Get the labels of the selected K entries.
7. If regression, return the mean of the K labels.
8. If classification, return the mode of the K labels.

For our distance measure, we will choose the **Euclidean distance** defined by the following equation:

$$
d(p, q) = \sqrt{(p - q)^{T} (p - q)}
$$

### Choosing the right value for K
To select the K that’s right for your data, we run the KNN algorithm several times with different values of K and choose the K that reduces the number of errors we encounter while maintaining the algorithm’s ability to accurately make predictions when it’s given data it hasn’t seen before.

Here are some things to keep in mind:

1. As we decrease the value of K to 1, our predictions become less stable. Just think for a minute, imagine K=1 and we have a query point surrounded by several reds and one green (I’m thinking about the top left corner of the colored plot above), but the green is the single nearest neighbor. Reasonably, we would think the query point is most likely red, but because K=1, KNN incorrectly predicts that the query point is green.
2. Inversely, as we increase the value of K, our predictions become more stable due to majority voting / averaging, and thus, more likely to make more accurate predictions (up to a certain point). Eventually, we begin to witness an increasing number of errors. It is at this point we know we have pushed the value of K too far.
3. In cases where we are taking a majority vote (e.g. picking the mode in a classification problem) among labels, we usually make K an odd number to have a tiebreaker.

### Then how to select the optimal K value?
- There are no pre-defined statistical methods to find the most favorable value of K.
- Initialize a random K value and start computing.
- Choosing a small value of K leads to unstable decision boundaries.
- The substantial K value is better for classification as it leads to smoothening the decision boundaries.
- Derive a plot between error rate and K denoting values in a defined range. Choose the K value having the minimum error rate. 

## Dataset
For the dataset, I am using the [Palmer Penguin dataset](https://github.com/yw110-1/INDE-577/tree/main/Data).

## Reference
Dr. Davila's Github, https://github.com/RandyRDavila/Data_Science_and_Machine_Learning_Spring_2022
