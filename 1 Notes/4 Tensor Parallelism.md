Despite the clever tricks we've learnt in [[2 Training on One GPU]] to manage activation memory in GPUs, can we do better? Yes, through tensor parallelism, let's introduce the idea with some simple matrix multiplication.


Activations are the intermediate tensors produced during the forward pass that must be kept around so back propagation can use them later.

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

So here, each GPU could store one column block of B, compute its own output block, and then the outputs are concatenated.


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

For an activation tensor of shape $(b, s, h)$, the hidden dimension h is partitioned across devices. So with n GPUs, each GPU typically holds only a hidden shard of size $(b, s, h/n)$, computes its local part of the layer, and then synchronizes with the others when the full result is needed.

- $b$ = batch size
- $s$ = sequence length (number of tokens)
- $h$ = hidden dimension

---
## Example: MLP Layer In LLMs

![[Pasted image 20260328112624.png]]

Suppose we use tensor parallel size $=4$, then we split the first weight matrix $W$ by columns:

$W = [W_1 ; W_2 ; W_3 ; W_4]$

Where each shard has shape $W_i \in \mathbb{R}^{h \times h}$.

Each GPU gets:
- the same input activation $X$
- one shard of $W$

So each GPU computes:

$Z_i = XW_i \quad Z_i \in \mathbb{R}^{b \times s \times h}$

Then each GPU applies the activation locally:

$A_i = \phi(Z_i)$

So after the first linear layer and activation, each GPU holds only its own slice of the expanded hidden state. Conceptually, the full expanded activation is:

$A = [A_1 ; A_2 ; A_3 ; A_4] \in \mathbb{R}^{b \times s \times 4h}$

But it remains sharded across GPUs.

Now we want to project back down from $4h$ to $h$, so we split the second weight matrix $V^\top$ by rows:

$V^\top = \begin{bmatrix} V_1^\top \ V_2^\top \ V_3^\top \ V_4^\top \end{bmatrix}$

where each shard has shape $V_i^\top \in \mathbb{R}^{h \times h}$.

Each GPU uses its local activation shard $A_i$ with its matching row shard $V_i^\top$:

$Y_i = A_i V_i^\top \quad Y_i \in \mathbb{R}^{b \times s \times h}$.

They are partial contributions to the same final output. So we add them:

$Y = \sum_{i=1}^{4} Y_i$

This final summation is done with an all-reduce across the GPUs.


Just so we are not confused... let's make sure we know what each GPU stores. With tensor parallel size $=4$:
- every GPU sees the full input $X$ of shape $(b,s,h)$
- GPU 1 stores $W_1$ and $V_1^\top$
- GPU 2 stores $W_2$ and $V_2^\top$
- GPU 3 stores $W_3$ and $V_3^\top$
- GPU 4 stores $W_4$ and $V_4^\top$

---
# Tensor Parallelism Tradeoff

Tensor parallelism reduces memory pressure by splitting weights and intermediate activations across GPUs. However, once we shard these tensors, each GPU holds only a partial result rather than the full activation.

In the MLP case shown here, each GPU can compute FC1 → Activation → FC2 locally, but after FC2 the outputs are still partial contributions to the same final tensor. 

At certain points, the model must stop and synchronize those partial results before it can continue, for things like LayerNorm, here's a quick recap:

If a hidden representation is:

$x = \begin{bmatrix} x_1 & x_2 & \cdots & x_h \end{bmatrix} \in \mathbb{R}^{1\times h}$

$\mu = \frac{1}{h}\sum_{i=1}^{h} x_i$

$\sigma^2 = \frac{1}{h}\sum_{i=1}^{h}(x_i - \mu)^2$

$\operatorname{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$

To compute the true $\mu$ and $\sigma^2$, LayerNorm needs access to all coordinates of that token’s hidden state.

An all-reduce is therefore required to combine them before the next step, so the following LayerNorm can proceed. 
- This adds communication overhead directly to the forward pass.
- And also... defeats the point of tensor parallelism, as we still have to combine the entire input of $(b, s, h)$ per GPU

![[HF_ULTRASCALE_PLAYBOOK 11.jpg]]

[[HF_ULTRASCALE_PLAYBOOK.pdf#page=83&rect=69,495,359,556&color=yellow|HF_ULTRASCALE_PLAYBOOK, p.83]]

As tensor parallelism increases, we can fit larger batch sizes, but throughput per GPU decreases because the model incurs more communication overhead from synchronization across GPUs, especially all-reduce operations to combine partial results and all-gather operations to reconstruct full activations (for things like layer norm).

In practice, the communication overhead of tensor parallelism becomes particularly noticeable as we scale beyond 8 GPUs.

![[HF_ULTRASCALE_PLAYBOOK 10.jpg]]

[[HF_ULTRASCALE_PLAYBOOK.pdf#page=85&rect=69,453,361,557&color=yellow|HF_ULTRASCALE_PLAYBOOK, p.85]]

Is there a way that we can also parallel the operations like layer norm and drop out which require the full activations? Yes! That's the next part.

---
## Sequence parallelism

The idea of sequence parallelism is simple, we don't want to do all reduces to hold the full duplicate copy of the full output tensor, which is of shape  $(b, s, h)$, where:

- $b$ = batch size
- $s$ = sequence length (number of tokens)
- $h$ = hidden dimension

But recall, layer norm and dropout can be calculated independently per token's forward pass, token #1's LayerNorm calculation does not care about Token #2's data, so we can split the data among the sequence dimension.

Instead of giving every GPU the full $(b, s, h)$ tensor, Sequence Parallelism slices the tensor along the sequence dimension ($s$).

If you have n GPUs:
- GPU 0 gets the first half of the tokens: Shape is $(b, s/n, h)$.
- GPU 1 gets the second half of the tokens: Shape is $(b, s/n, h)$.
- GPU n gets the second half of the tokens: Shape is $(b, s/n, h)$.

Because each GPU has the _full_ $h$ dimension for its specific batch of tokens, it can compute LayerNorm and Dropout  independently with zero communication.


So now, we can see that we can use both tensor parallel and sequence parallel together, as different operations require different dimensions to be intact, we route them to the appropriate parallelism strategy:

- SP Regions (LayerNorm & Dropout)
- TP Regions (Self-Attention & Linear/MLP layers)

![[120 CS/123 AI/3 NLP/4 Ultra Scale Playbook/1 Notes/attachments/HF_ULTRASCALE_PLAYBOOK 12.jpg|500]]

[[HF_ULTRASCALE_PLAYBOOK.pdf#page=88&rect=38,247,326,556|HF_ULTRASCALE_PLAYBOOK, p.88]]

But this raises another question, how do we transition from TP to SP? Or SP to TP? Well, it's easiest to see when we have an example, assume that we have n GPUs:

From SP to TP:
- each GPU starts with only its chunk of the sequence, of shape $(b, s/n, h)$. We do an all-gather across the sequence dimension, so every GPU receives the missing token chunks and reconstructs the full activation tensor $(b, s, h)$. Now the tensor-parallel linear layer can run.

From TP to SP:
- after the tensor-parallel computation, GPUs hold partial outputs that must be combined. We use reduce-scatter: the reduce part sums the partial results across GPUs, and the scatter part immediately splits the final tensor back across the sequence dimension. This gives each GPU a shard of shape $(b, s/n, h)$ again.

In short:
- SP to TP uses all-gather.
- TP to SP uses reduce-scatter.

![[120 CS/123 AI/3 NLP/4 Ultra Scale Playbook/1 Notes/attachments/HF_ULTRASCALE_PLAYBOOK 13.jpg|300]]

[[HF_ULTRASCALE_PLAYBOOK.pdf#page=91&rect=135,254,304,552|HF_ULTRASCALE_PLAYBOOK, p.91]]

By switching between TP (sliced along $h$) and SP (sliced along $s$), the model never has to hold the massive, full $(b, s, h)$ activation tensor in memory. The maximum activation size per GPU drops to $\frac{b \cdot s \cdot h}{TP}$, allowing you to train on much longer context windows.

![[120 CS/123 AI/3 NLP/4 Ultra Scale Playbook/1 Notes/attachments/HF_ULTRASCALE_PLAYBOOK 14.jpg|500]]

[[HF_ULTRASCALE_PLAYBOOK.pdf#page=92&rect=37,388,326,555|HF_ULTRASCALE_PLAYBOOK, p.92]]

And this is how the overhead of computations look like in the forward pass.

![[120 CS/123 AI/3 NLP/4 Ultra Scale Playbook/1 Notes/attachments/HF_ULTRASCALE_PLAYBOOK 15.jpg]]

[[HF_ULTRASCALE_PLAYBOOK.pdf#page=94&rect=36,493,328,557|HF_ULTRASCALE_PLAYBOOK, p.94]]

