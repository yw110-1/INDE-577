# Gradient Descent
In this sub-repository, I will introduce a general continuous optimization technique called gradient descent. Gradient Descent is an iterative first-order optimization algorithm used to find a local minimum/maximum of a given function. This method is commonly used in machine learning and deep learning to minimise a cost/loss function (e.g. in a linear regression).

## Algorithm
<p align="center">
<img src="https://github.com/yw110-1/INDE-577/blob/main/Supervised%20Learning/Perceptron/image/gradient.png" alt="gradient" width="500"/>
</p>

The general idea behind gradient descent is to use the gradient (the derivative for single variable functions) to provide a direction to explore (this means gradient descent is a first-order method). For example, with our function $f$ and initial guess $w_0 = 5$, suppose we are able to calculate the value of the gradient (the derivative) of $f(w)$ at $w_0 = 5$. This numerical value will give us the slope of the tangent line to $f(w)$ at $w_0$. Note that $f'(w) = 2(w - 2)$.

### Gradient Descent Procedure

- Initialize values for the coefficients for the function. These could be 0 or a small random value. So, we make coefficients = 0
- The cost of the function is evaluated by plugging the coefficients into the function i.e., we do cost = f(coefficients)
- The derivative of the cost function is calculated. We need to know the slope so that we know the direction (sign) to move the coefficient values in order to get a lower cost on the next iteration. So, we calculate, change = derivative(cost)
- Now that we know the downhill direction from the derivative, we can now update the coefficient values. Specify a learning rate that controls how much the coefficients can change on each update. So, we do, coefficient = coefficient — (learning rate * change)
- We repeat this process until the cost is 0 or close to zero.

### How far should we move?
The value of how far to move in the opposite sign of the derivative of $f(w)$ at $w_0 = 5$ is called the learning rate (Nocedal & Wright call this hyperparameter the step length), and is typically denoted by $\alpha$. The process of multiplying the derivative of $f(w)$ at $w_0 = 5$ by the learning rate and forming a new choice of $w$ by subtracting this quantity from $w_0$ is called gradient descent. For example, we may apply gradient descent at $w_0$ and form a new $w$, say $w_1$, with the following update:

$$ w_{n+1} = w_n - \alpha f'(w_n) $$

The choice of $\alpha$ in machine learning is typically found by experimentation, though more sophesticated techniques are available, such as line-search and trust-region methods (again see Nocedal & Wright). 

### Minimizing Functions of Several Variables
All of the ideas above naturally generalize to functions of several variables when substituting the gradient for the single variable derivative. Before discussing this notion, we emphasize the general uncrontrained optimization problem:

$$ \min_{w\in \mathbb{R^n}} f(w)$$

For instructional purposes we next give a specific solution to this problem by focusing on a function of two variables, though all notions covered extend to functions of an arbitrary and finite number of variables. In a general manor, gradient descent can now be defined in a meaningful sense:

$$ w \leftarrow w - \alpha \nabla f(w) $$

With this multivariable gradient descent defined we now consider the multivariable function $f(w_0, w_1) = w_0^2 + w_1^2 + 1$ which has an obvious minimum at the vector $\mathbf{w} = [0.0, 0.0]^T$. To visualize this function run the following code in the cell below.

## Reference
1. Dr. Davila's Github, https://github.com/RandyRDavila/Data_Science_and_Machine_Learning_Spring_2022
2. Rekha M, The Ascent of Gradient Descent, https://blog.clairvoyantsoft.com/the-ascent-of-gradient-descent-23356390836f
