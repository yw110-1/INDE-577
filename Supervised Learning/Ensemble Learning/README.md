# Ensemble Learning
In this sub-repository, I will introduce ensemble learning methods. Ensemble methods are machine learning methods that aggregate the predictions of a group of base learners in order to form a single learning model. The methods I will include are **Bagging Classification** and **Random Forests**.

## Algorithm
### Bagging
Bagging, also known as Bootstrap aggregating, is an ensemble learning technique that helps to improve the performance and accuracy of machine learning algorithms. It is used to deal with bias-variance trade-offs and reduces the variance of a prediction model. Bagging avoids overfitting of data and is used for both regression and classification models, specifically for decision tree algorithms.

The steps for bagging classifier includes: 
- Consider there are n observations and m features in the training set. You need to select a random sample from the training dataset without replacement
- A subset of m features is chosen randomly to create a model using sample observations
- The feature offering the best split out of the lot is used to split the nodes
- The tree is grown, so you have the best root nodes
- The above steps are repeated n times. It aggregates the output of individual decision trees to give the best prediction

<p align="center">
<img src="https://github.com/yw110-1/INDE-577/blob/main/Supervised%20Learning/Perceptron/image/bagging.png" alt="bagging" width="700"/>
</p>

### Random Forest
Random forest is a Supervised Machine Learning Algorithm that is used widely in Classification and Regression problems. It builds decision trees on different samples and takes their majority vote for classification and average in case of regression.

Steps involved in random forest algorithm:
- In Random forest n number of random records are taken from the data set having k number of records.
- Individual decision trees are constructed for each sample.
- Each decision tree will generate an output.
- Final output is considered based on Majority Voting or Averaging for Classification and regression respectively.

<p align="center">
<img src="https://github.com/yw110-1/INDE-577/blob/main/Supervised%20Learning/Perceptron/image/randomforest.jpg" alt="randomforest" width="700"/>
</p>

## Dataset
The dataset I am using in ensemble learning is [Load_wine dataset](https://github.com/yw110-1/INDE-577/tree/main/Data).

## References
1. Dr. Davila's Github, https://github.com/RandyRDavila/Data_Science_and_Machine_Learning_Spring_2022
2. Biswal, Avijeet, Bagging in Machine Learning: Steps to Perform and Its Advantages, https://www.simplilearn.com/tutorials/machine-learning-tutorial/bagging-in-machine-learning
3. Understanding Random Forest, https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/
