# Principle Component Analysis
In this sub-repository I will introduce the Principle Component Analysis, or PCA. PCA is a dimensionality-reduction method that is often used to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information in the large set.

## Algorithm
- Write N datapoints $\\mathbf{x_i}$ = ( $\\mathbf{x_{1i}}, \\mathbf{x_{2i}}, ... \\mathbf{x_{Mi}}$) as row vectors
- Put these vectors into a matrix $\\mathbf{X}$
- Centre the data by stracting off the mean of each column, putting it into matrix $\\mathbf{B}$
- Compute the covariance matrix $\\mathbf{C} = \frac{1}{N} \\mathbf{B}^T \\mathbf{B}$
- Compute the eigenvalues and eigenvectors of $\\mathbf{C}$, so $\\mathbf{V}^{-1} \\mathbf{C} \\mathbf{V} = D$, where $\\mathbf{V}$ holds the eigenvectors of $\\mathbf{C}$ and $\\mathbf{D}$ is the M*M diagonal eigenvalue matrix
- Sort the columns of $\\mathbf{D}$ into order of decreasing eigenvalues, and apply the same order to the columns of V
- Reject those with eigenvalue less than some $\eta$, leaving L dimensions in the data



## References
1. Builtin, A Step-by-Step Explanation of Principal Component Analysis (PCA), https://builtin.com/data-science/step-step-explanation-principal-component-analysis.
2. Bamboos Consulting, Naive Principal Component Analysis, https://www.bamboos-consulting.com/blog/naive-principal-component-analysis/.
