Despite the clever tricks we've learnt in [[2 Training on One GPU]] to manage activation memory in GPUs, can we do better and somehow shard these as well? Yes, through tensor parallelism, let's introduce the idea with some simple matrix multiplication.

Take:

$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \qquad B = \begin{bmatrix} 10 & 30 \\ 20 & 40 \end{bmatrix}$

Then:

$AB = \begin{bmatrix} 1\cdot 10 + 2\cdot 20 & 1\cdot 30 + 2\cdot 40 \\ 3\cdot 10 + 4\cdot 20 & 3\cdot 30 + 4\cdot 40 \end{bmatrix} = \begin{bmatrix} 50 & 110 \\ 110 & 250 \end{bmatrix}$

This is basic, simple linear algebra, but notice how we can parallelize this process in 2 ways.
  

1. Column-wise parallelism

Split B by columns:

$B = \left[ B_1 \;\; B_2 \right] \quad\text{where}\quad B_1 = \begin{bmatrix} 10 \\ 20 \end{bmatrix}, \qquad B_2 = \begin{bmatrix} 30 \\ 40 \end{bmatrix}$

Now compute each piece separately:

$AB_1 = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 10 \\ 20 \end{bmatrix} = \begin{bmatrix} 50 \\ 110 \end{bmatrix}$

$AB_2 = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 30 \\ 40 \end{bmatrix} = \begin{bmatrix} 110 \\ 250 \end{bmatrix}$

Then concatenate the results:

$AB = [AB_1 \;\; AB_2] = \begin{bmatrix} 50 & 110 \\ 110 & 250 \end{bmatrix}$

So here, each GPU could store one column block of B, compute its own output block, and then the outputs are combined side by side.


2. Row-wise parallelism

Now split A by columns and B by matching rows:

$A = [A_1 \;\; A_2] \quad\text{where}\quad A_1 = \begin{bmatrix} 1 \\ 3 \end{bmatrix}, \qquad A_2 = \begin{bmatrix} 2 \\ 4 \end{bmatrix}$

$B = \begin{bmatrix} B_1 \\ B_2 \end{bmatrix} \quad\text{where}\quad B_1 = \begin{bmatrix} 10 & 30 \end{bmatrix}, \qquad B_2 = \begin{bmatrix} 20 & 40 \end{bmatrix}$

Now compute the partial products:

$A_1B_1 = \begin{bmatrix} 1 \\ 3 \end{bmatrix} \begin{bmatrix} 10 & 30 \end{bmatrix} = \begin{bmatrix} 10 & 30 \\ 30 & 90 \end{bmatrix}$

$A_2B_2 = \begin{bmatrix} 2 \\ 4 \end{bmatrix} \begin{bmatrix} 20 & 40 \end{bmatrix} = \begin{bmatrix} 40 & 80 \\ 80 & 160 \end{bmatrix}$

Then add them:

$AB = A_1B_1 + A_2B_2 = \begin{bmatrix} 10 & 30 \\ 30 & 90 \end{bmatrix} + \begin{bmatrix} 40 & 80 \\ 80 & 160 \end{bmatrix} = \begin{bmatrix} 50 & 110 \\ 110 & 250 \end{bmatrix}$

So here, each GPU computes a partial contribution to the same final output, and then those partial outputs are summed together.


In summary, for **column-wise parallelism**, we split the weight matrix by columns:

$A[B_1\;B_2\;\cdots\;B_n] = [AB_1\;AB_2\;\cdots\;AB_n]$

Each GPU computes one output block, and the final result is formed by concatenating those blocks.

For **row-wise parallelism**, we split the input and weight in matching pieces:

$[A_1\;A_2\;\cdots\;A_n] \begin{bmatrix} B_1 \\ B_2 \\ \vdots \\ B_n \end{bmatrix} = \sum_{i=1}^{n} A_i B_i$

Each GPU computes one partial result, and the final result is formed by summing them.


On the computer, this turns into 2 different operations:

- Column-wise parallelism:
	1. broadcast the matrix to each GPU
	2. compute the matrix multiplication
	3. all gather and concat the matrixes together

- Row-wise parallelism:
	1. scatter the 
	2. compute the matrix multiplication
	3. all reduce and add the matrixes together element wise

---

## Transformer Block Parallelism

We understand 
