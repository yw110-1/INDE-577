# Linear Regression
In this sub-repository, I will introduce the single neuron linear regression model. Linear regression analysis is used to predict the value of a variable based on the value of another variable.

## Algorithm
This specific case of regression assumes that the target values in $\mathcal{Y}$ are approximated by a linear function of the associated feature vectors. That is, the optimal target function $f:\mathcal{X} \rightarrow \mathcal{Y}$ is assumed the be roughly a linear function. 

### Supervised Machine Learning
<p align="center">
<img src="https://github.com/yw110-1/INDE-577/blob/main/Supervised%20Learning/Perceptron/image/supervised_learning.png" alt="supervised_learning" width="500"/>
</p>

In regression, machine learning models are given labeled data $\mathcal{D} = \{(\mathbf{x}^1, y^1), \dots, (\mathbf{x}^N, y^N)\}$, where the feature vectors satisfy $\mathbf{x}^{(i)} \in \mathbb{R}$ and the target labels satify $y^{(i)} \in \mathbb{R}$. Thus, this supervised learning task seeks to predict real valued target labels. This is different from classification (such as the perceptron single neuron model) as the following figure suggests.

<p align="center">
<img src="https://github.com/yw110-1/INDE-577/blob/main/Supervised%20Learning/Perceptron/image/Regression_VS_Classification.png" alt="Regression_VS_Classification" width="500"/>
</p>

### Linear Regression Single Neuron Model
There exists a general machine learning model for supervised learning. In this model the optimal target function $f:\mathcal{X} \rightarrow \mathcal{Y}$ assigns the correct labels to every possible feature measurement. Next recall that our goal is to find a reasonable hypthesis $h:\mathcal{X} \rightarrow \mathcal{Y}$, which approximates the target function $f$. The general ML model is shown as below.

<p align="center">
<img src="https://github.com/yw110-1/INDE-577/blob/main/Supervised%20Learning/Perceptron/image/General_ML_Model.png" alt="General_ML_Model" width="500"/>
</p>

Because we are assuming the target function $f:\mathcal{X} \rightarrow \mathcal{Y}$ is a linear function of the input features, and because we know single neuron models are good function approximators, we next build a single neuron model with a linear-activation activation function. Furthermore, in this model we choose the mean-sqaured error cost function:

$$ C(\mathbf{w}, b) = \frac{1}{2N}\sum_{i=1}^{N}\Big(\hat{y}^{(i)} - y^{(i)}\Big)^2. $$

With our specific case of linear regression on the setosa iris dataset with a single feature measurment taken from sepal length data, we next construct a single neuron model with a linear activation function and the mean-sqaured error cost function as depicted in the figure below.

<p align="center">
<img src="https://github.com/yw110-1/INDE-577/blob/main/Supervised%20Learning/Perceptron/image/regression_neuron.png" alt="regression_neuron" width="500"/>
</p>

### Minimize the the Cost Function $C(w_1, b)$
Before defining a custom SingleNeuron class, we first need first discuss how to minimize the neurons cost function. More specifically, we wish to solve the following optimization problem:

$$ \min_{w_1, b}C(w_1, b) $$

Since $C(w_1, b)$ is a differentiable function of both $w_1$ and $b$, we may attempt to solve this minimization problem by applying the gradient descent algorithm:

$$ w_1 \leftarrow w_1 - \alpha \frac{\partial C}{\partial w_1} $$

$$ b \leftarrow b - \alpha \frac{\partial C}{\partial b} $$

### - Finding the Partial Derivatives of $C(w_1, b)$
In order to implement the gradient descent method we first need to understand how the partial derivatives of $C(w_1, b)$ are calculated over the training data at hand. With this in mind, suppose for now that we are calculating the mean-sqaured error cost function on a single example example of data, i.e., $N = 1$. For this single example we observe that the mean-sqaured error cost function becomes:

$$ C(w, b; \mathbf{x}^{(i)}, y^{(i)}) = \frac{1}{2}\Big(\hat{y}^{(i)} - y^{(i)}\Big)^2. $$

In the case of a linear activation function, it is important to note that $\hat{y}^{(i)}$ is a very simple function of both $w_1$ and $b$. More specifically, we observe:

$$ \hat{y}^{(i)} = a = z = w_1x^{(i)} + b. $$

Thus, we may rewrite our neuron cost function with a single observation:

$$ C(w, b; \mathbf{x}^{(i)}, y^{(i)}) = \frac{1}{2}\Big(w_1x^{(i)} + b - y^{(i)}\Big)^2. $$

With this equation, we can calculate $\partial C/ \partial w_1$ and $\partial C/ \partial b$ easily by applying the chain rule (click for a quick refresher on the concept). The resulting partial derivatives with respect to $w_1$ and $b$ shown by the following equations:

$$\frac{\partial C(w_1, b; \mathbf{x}^{(i)}, y^{(i)})}{\partial w_1} = (w_1x^{(i)} + b - y^{(i)})x^{(i)} = (\hat{y}^{(i)} - y^{(i)})x^{(i)}$$

$$\frac{\partial C(w_1, b; \mathbf{x}^{(i)}, y^{(i)})}{\partial b} = (w_1x^{(i)} + b - y^{(i)}) = (\hat{y}^{(i)} - y^{(i)})$$

Understanding the different ways in which we may calculate the partial derivatives of our cost function is essential in applying any first-order minimization technique on the cost function $C(w_1, b)$. With what follows we discuss two of the three fundamental methods used to accomplish this goal.

### - Different Flavors of First-Order Minimization
When considering a single instance of data, we easily calculated $\frac{\partial C}{\partial w_1}$ and $\frac{\partial C}{\partial b}$ by applying the chain-rule. This notion can now be extended to all data used in training by summing the gradients calculated at entry of data. We will refer to this process as calculating the full gradient (or full partial derivatives) with respect to the training data:

$$\frac{\partial C(w_1, b; \mathbf{X}, y)}{\partial w_1} = \frac{1}{N}\sum_{i=1}^{N}\Big(\hat{y}^{(i)} - y^{(i)}\Big)x^{(i)}$$

$$\frac{\partial C(w_1, b; \mathbf{X}, y)}{\partial b} = \frac{1}{N}\sum_{i=1}^{N}\Big(\hat{y}^{(i)} - y^{(i)}\Big)$$

Calculating the full gradient with respect to all training data and applying the gradient descent algorithm is called batch gradient descent.

Flavor 1. Batch Gradient Descent Algorithm:

For each epoch do
Calculate the full gradient by finding $\frac{\partial C(w_1, b; \mathbf{X}, y)}{\partial w_1}$ and $\frac{\partial C(w_1, b; \mathbf{X}, y)}{\partial b}$.
$$w \leftarrow w - \alpha \frac{\partial C(w_1, b; \mathbf{X}, y)}{\partial w_1}$$

$$b \leftarrow b - \alpha \frac{\partial C(w_1, b; \mathbf{X}, y)}{\partial b}$$

Applying batch gradient descent will work. However, this method can be very slow and use a lot of memory, especially when the number of training data is very large (possibly millions). More importantly, batch gradient descent is not necessary to find local minima.

The most common way work around for this problem is to update $w_1$ and $b$ by calculating the gradient with respect to one entry of data at a time. This technique is called stochastic gradient descent and is one of the primary tools in training deep neural networks and simple single neuron models.

Flavor 2. Stochastic Gradient Descent Algorithm:

For each epoch do
For $i = 1, \dots, N$ do
Calculate $\frac{\partial C(w_1, b; \mathbf{x}^{(i)}, y^{(i)})}{\partial w_1}$ and $\frac{\partial C(\partial C(w_1, b; \mathbf{x}^{(i)}, y^{(i)}))}{\partial b}$.

$$w \leftarrow w - \alpha \frac{\partial C(w_1, b; \mathbf{x}^{(i)}, y^{(i)})}{\partial w_1}$$

$$b \leftarrow b - \alpha \frac{\partial C(w_1, b; \mathbf{x}^{(i)}, y^{(i)})}{\partial b}$$


## Dataset
The dataset I am using is [Palmer Penguin dataset](https://github.com/yw110-1/INDE-577/tree/main/Data).

## Reference
Dr. Davila's Github, https://github.com/RandyRDavila/Data_Science_and_Machine_Learning_Spring_2022
