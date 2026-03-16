Recall the fundamental steps of training a machine learning mode:
1. forward pass data from the model
2. backward pass to compute gradients
3. optimizer step to update model parameters

In this image, boxes can be seen as successive layers inside a model.

![[HF_ULTRASCALE_PLAYBOOK 1.jpg|500]]

[[HF_ULTRASCALE_PLAYBOOK.pdf#page=20&rect=32,379,334,561|HF_ULTRASCALE_PLAYBOOK, p.20]]

## Memory in Transformers

When training a neural network model, we store several items in memory:
- Model weights
- Model gradients
- Optimizer states
- Activations needed to compute the gradients

The number of parameters in a LLM follow this formula:

$N = h \cdot v + L \cdot (12 \cdot h^2 + 13 \cdot h) + 2 \cdot h$

-  $h$ is the hidden dimension
- $v$ the vocabulary size
- $L$ the number of layers.

How this formula came to be is further explained [here](https://michaelwornow.net/2024/01/18/counting-params-in-transformer).

Each parameter is just a number, and that number needs to be stored somewhere. So the memory required for your model's parameters is simply:

$\text{Memory}_{\text{params}} = N \times \text{bytes per parameter}$

Gradients are simply values representing how much to change each parameter, so they take up the same amount of memory as the parameters themselves. The Adam optimizer additionally stores two values per parameter, the momentum and variance.

So in total, it looks something like this for when we use full precision, float 32 (4 bytes) to store the relevant training data for a model.

- $m_{\text{params}} = 4 \times N$
- $m_{\text{grad}} = 4 \times N$
- $m_{\text{opt}} = (4 + 4) \times N$


Lastly for activation values (saved for computing gradients in back propagation), they follow a bit of a weird formula:

 $m_{\text{activations}} = L \cdot seq \cdot bs \cdot h \cdot \left(34 + \frac{5 \cdot n_{\text{heads}} \cdot seq}{h}\right)$

- $L$ — number of layers
- $seq$ — sequence length
- $bs$ — batch size in samples
- $h$ — hidden dimension of the model
- $n_{\text{heads}}$ — number of attention heads

Again, how we got here can be derived from Nvidia's paper on re-computation [here](https://arxiv.org/abs/2205.05198).

Unlike parameter memory which is fixed by the model architecture, activation memory scales with your training hyper parameters. Larger batch sizes and longer sequences mean more data processed in parallel, and therefore more activation values to store for back propagation.

Now that we have a sense of how much memory each training component uses, we can compare how they scale as models grow and sequence lengths increase.

![[HF_ULTRASCALE_PLAYBOOK 2.jpg]]

[[HF_ULTRASCALE_PLAYBOOK.pdf#page=30&rect=38,400,326,555|HF_ULTRASCALE_PLAYBOOK, p.30]]

The graphs show a clear pattern: for short sequences or small batch sizes, activation memory is almost negligible. But around 2–4k tokens, it begins to grow substantially, while all other parts remain constant throughout.

So... what to do?

---
## Activation Re-computation

Basically, instead of storing activation values, just recompute them during the backward pass, we've shown an example of this in [[5 GPUs#^1e02f6|CS336, 5 GPUs - Recomputation]]:

![[5 GPUs#3. Re-computation 1e02f6]]


Now, what's shown above is _Full_ activation recompilation, meaning we store 0 activation values and recalculate them all during back propagation.

However, this means doubling the time it takes for forward passes, as we are doing 2 entire forward passes, 1 in the normal forward pass, and 1 in the backward pass.
- This roughly increases total compute time by 30-40%

We can do better than that! What if we save 'some' activation values in memory so we don't recompute everything in the forward pass from scratch? That's _Selective_ activation recompilation.
- what activations to select to be held in memory is a whole other detail to study by itself, which is also in Nvidia's paper on re-computation.
- this method achieved a 70% memory reduction at a 2.7% compute time increase for a GPT-3 model.

![[HF_ULTRASCALE_PLAYBOOK 3.jpg|500]]

[[HF_ULTRASCALE_PLAYBOOK.pdf#page=32&rect=26,62,335,564|HF_ULTRASCALE_PLAYBOOK, p.32]]


> [!PDF|yellow] [[HF_ULTRASCALE_PLAYBOOK.pdf#page=33&selection=14,0,22,71&color=yellow|HF_ULTRASCALE_PLAYBOOK, p.33]]
> > Most training frameworks these days use FlashAttention (covered further later in the book), which natively integrates activation recomputation in its optimization strategy by recomputing attention scores and matrices in the backward pass instead of storing them. Thus, most people using FlashAttention are already making use of selective recomputation.

So right now, we've broken the linear relationship between activation storage & sequence length, but what about activation storage & batch size? That's what we'll explore next.

## Gradient Accumulation

Not some crazy idea, think of it as stochastic gradient descent, but we take the average of $N$ gradients before updating the parameters with the optimizer.

With gradient accumulation, the batch size can be computed as follows:

$bs = gbs =  mbs \times grad\_acc$

- $bs$ — batch size
- $gbs$ — global batch size
- $mbs$ — micro-batch size
- $grad\_acc$ — number of gradient accumulation steps

How does this help us save memory, you ask? Well, if your global batch size is 1024, and you tried to run the whole thing at once, you'd need to store activations for all 1024 samples simultaneously.

 $g_i = \nabla_\theta \ell_i$
 $g = \frac{1}{1024}\sum_{i=1}^{1024} g_i$

- $\ell_i$ — the loss for sample i
- $\theta$ — the model parameters.
- $g_i$ — the gradient contribution from sample $i$
- $g$ — the gradient used for the optimizer step over the full batch (1024 samples in this case)

Now consider gradient accumulation with mbs = 32 and grad_acc = 32. The effective batch size is still:

$\text{gbs} = \text{mbs} \times \text{grad\_acc} = 32 \times 32 = 1024$

But instead of processing all 1024 samples at once, we split them into 32 micro-batches of size 32. For micro-batch $k$, define its gradient as

$g^{(k)} = \frac{1}{32}\sum_{j=1}^{32} g_{k,j}$

- $g_{k,j}$ — the gradient contribution of the $j$-th sample inside micro-batch $k$
- $g^{(k)}$ — the average gradient of micro-batch $k$

Then the overall gradient across all 32 micro-batches is:

$g = \frac{1}{32}\sum_{k=1}^{32} g^{(k)}$

Expanding that gives:

$g = \frac{1}{32}\sum_{k=1}^{32}\left(\frac{1}{32}\sum_{j=1}^{32} g_{k,j}\right) = \frac{1}{1024}\sum_{i=1}^{1024} g_i$

So even though we only process 32 samples at a time, after accumulating across 32 such micro-batches, we recover the same gradient we would have gotten from a single batch of size 1024, as the accumulated gradient is mathematically equivalent to the gradient you would have gotten from one true batch of size 1024.

With gradient accumulation, the process looks like this instead:
1. run forward on 32 samples
2. store activations for only those 32 samples
3. run backward on those 32 samples
4. add their gradients into the running gradient buffer
5. repeat 32 times
6. perform one optimizer step

![[HF_ULTRASCALE_PLAYBOOK 1 1.jpg|500]]

[[HF_ULTRASCALE_PLAYBOOK.pdf#page=36&rect=32,337,338,561&color=yellow|HF_ULTRASCALE_PLAYBOOK, p.36]]

However, these micro-batches mean a lot more forward/backward passes before taking a single optimizer step. While reducing memory usage for activations, it introduces overhead from repeated kernel launches on the GPU;

But... since the forward/backward computations for different micro-batches are independent, why not run them on multiple GPUs at the same time instead of one after another on a single GPU? Welcome to data parallelism in the next chapter.


