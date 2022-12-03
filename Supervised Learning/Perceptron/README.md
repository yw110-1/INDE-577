# Perceptron
In this sub-repository we will introduced a specific type of single neuron model called the perceptron.

## Algorithm
A perceptron is the simplest neural network, one that is comprised of just one neuron. 
- Input: A sequence of training samples ( $\\mathbf{x_1}$, $y_1$), ( $\\mathbf{x_2}$, $y_2$), ...
- First we initialize $\\mathbf{w_0} = 0$
- For each training example ( $\\mathbf{x_i}$, $y_i$):
  - Predict $\\widehat{y_i} = \\phi( \\mathbf{w}_t^{\\top} \\mathbf{x_i} )$
  - If $y_i$ is not equal to $\\widehat{y}$:
    - Update  $\\mathbf{w}_{t+1} = \\mathbf{w}_t + \\eta ( y_i - \\widehat{y_i}) \\mathbf{x_i}$
  - Repeat until iteration error is small
  - Return final weight vector $\\mathbf{w_i}$

The figure of perceptron with the sign activation function is as below:
<img src="https://github.com/yw110-1/INDE-577/blob/main/Supervised%20Learning/Perceptron/image/perceptron.png" alt="perceptron" width="600"/>

And the perception replace rule is shown as below:

<img src="https://github.com/yw110-1/INDE-577/blob/main/Supervised%20Learning/Perceptron/image/Update%20Rule.png" alt="perceptron" width="700"/>
