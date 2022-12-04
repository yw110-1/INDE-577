# Principle Component Analysis
In this sub-repository I will introduce the Principle Component Analysis, or PCA. PCA is a dimensionality-reduction method that is often used to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information in the large set. The goal of PCA is to reduce the dimensionality of the feature vectors used in training machine learning algorithms. 

## Algorithm
- Write N datapoints $\\mathbf{x_i}$ = ( $\\mathbf{x_{1i}}, \\mathbf{x_{2i}}, ... \\mathbf{x_{Mi}}$) as row vectors
- Put these vectors into a matrix $\\mathbf{X}$
- Centre the data by stracting off the mean of each column, putting it into matrix $\\mathbf{B}$
- Compute the covariance matrix $\\mathbf{C} = \frac{1}{N} \\mathbf{B}^T \\mathbf{B}$
- Compute the eigenvalues and eigenvectors of $\\mathbf{C}$, so $\\mathbf{V}^{-1} \\mathbf{C} \\mathbf{V} = D$, where $\\mathbf{V}$ holds the eigenvectors of $\\mathbf{C}$ and $\\mathbf{D}$ is the M*M diagonal eigenvalue matrix
- Sort the columns of $\\mathbf{D}$ into order of decreasing eigenvalues, and apply the same order to the columns of V
- Reject those with eigenvalue less than some $\eta$, leaving L dimensions in the data

### Steps for Principle Component Analysis in code:
1. **Standardize (center and scale) the data.** 

To center the data, we average each row by replacing the value $x$ by 

$$
x - \text{mean}
$$

Data values may have vastly different ranges, and so, to ensure that PCA is not selecting wrong directions in describing data variation, we also divide by the standard deviation. That is, we scale the data in each variable by finding the *z-scores*:

$$
z = \frac{x - \text{mean}}{\text{standard devation}}
$$

Finally, we form the $m\times n$ matrix $A$. 

2. **Compute the covariance or correlation matrix**:

$$
S = \frac{1}{n-1}AA^T
$$

If we are working with only centered data, the above matrix is the covariance matrix, and if we are working with scaled data, then $S$ is the correlation matrix. The entries on the diagonal are the variances (or correlations) for each variable and the off-diagonal entries are the covariances (or correlations) between two variables: positive covariance indicates that the variables are directly related (when one increases, the other increases as well), negative covariance indicates inverse relationship (when one increases, the other decreases). This matrix is symmetric of size $m \times m$, so its columns are of the same size as the columns of $A$.

3. **Find the eigenvalues and the orthonormal eigenvectors of $S$.** 

These eigenvectors are columns of the matrix $U$ in the singular value decomposition of $A$, up to the factor $n-1$. Further, we denote the eigenvalues by $\sigma_{i}^{2}$. This is equivalent to the **Singular Value Decomposition** of our shifted training set matrix $A$,

$$
A = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^{T}, 
$$

4. **Find the principal components.**

We arrange the eigenvalues found in the previous step in the decreasing order. The first principal component $PC_1$ is in the direction of the 1st eigenvector, the second principal component $PC_2$ is in the direction of the 2nd eigenvector, etc. The entries of each $PC_i$ are called *loading scores* and they tell us how the $PC_i$ is a linear combination of features.

5. **Reduce the dimension of the data.**

We project data points (i.e., columns of $A$) onto the selected principal components (i.e., several eigenvectors of $S$). By the Eckart-Young theorem we know that the line closest to the data points is in the direction of $PC_1$, etc (”closest” is in the sense of perpendicular least squares).

In addition, the total variance, which is the trace of $S$, is

$$
T = \text{trace}(S) = \frac{\sigma_{1}^{2} + \dots + \sigma_{m}^{2}}{n-1},
$$

and the $i$-th principle component $PC_i$ explains

$$
\frac{\sigma_{i}^{2}/(n-1)}{T} = \frac{\sigma_{i}^{2}}{\sigma_{1}^{2} + \dots + \sigma_{m}^{2}}
$$

of the total variation. We use a scree plot to graph the percentages of variation that each $PC_i$ accounts for. Also, the sum of squared distances from the points projected to $PC_i$ to the origin is the eigenvalue for $PC_i$ or the squared singular value $\sigma_{i}^{2}$.

To project the data contained in $A$ onto the first two principle component axis, we compute $A [PC_1 PC_2]$.



## Dataset
The dataset I am using is [Palmer Penguin dataset](https://github.com/yw110-1/INDE-577/tree/main/Data).

## References
1. Builtin, A Step-by-Step Explanation of Principal Component Analysis (PCA), https://builtin.com/data-science/step-step-explanation-principal-component-analysis.
2. Bamboos Consulting, Naive Principal Component Analysis, https://www.bamboos-consulting.com/blog/naive-principal-component-analysis/.
