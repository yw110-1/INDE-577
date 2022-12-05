# Decision Trees
In this sub-repository, I will intri=oduce the decision trees. Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise constant approximation.

## Algorithm
Decision trees tend to be the method of choice for predictive modeling because they are relatively easy to understand and are also very effective. The basic goal of a decision tree is to split a population of data into smaller segments. There are two stages to prediction. The first stage is training the model—this is where the tree is built, tested, and optimized by using an existing collection of data. In the second stage, you actually use the model to predict an unknown outcome.

Decision trees are constructed from only two elements — nodes and branches:

<p align="center">
<img src="https://github.com/yw110-1/INDE-577/blob/main/Supervised%20Learning/Perceptron/image/Decision_Tree.jpeg" alt="decisiontree" width="600"/>
</p>

The nodes shown above fall under the following types of nodes:

1. Root node — node at the top of the tree. This node acts as the input node for feature vectors in the model.
2. Decision nodes — nodes where the variables are evaluated. These nodes have arrows pointing to them and away from them
3. Leaf nodes — final nodes at which the prediction is made

## Dataset
The dataset I am using is [Palmer Penguin dataset](https://github.com/yw110-1/INDE-577/tree/main/Data).

## Reference
Dr. Davila's Github, https://github.com/RandyRDavila/Data_Science_and_Machine_Learning_Spring_2022
