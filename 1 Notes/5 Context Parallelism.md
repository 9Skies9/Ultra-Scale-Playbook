It does get confusing to question the difference between sequence parallelism and context parallelism, because both of them divide our input tensors on the dimension size of sequence in a input tensor of $(b, s, h)$

Sequence parallelism is a memory-saving layout strategy where we shard the input along the sequence dimension for operations that are independent across token positions. For layers like LayerNorm, dropout, and residual add, each token can be processed independently, so each GPU only needs its own local chunk of tokens rather than the full sequence.

Context parallelism extends this idea to the entire model, including the parts that normally still require the full sequence, especially attention. 

But... how do we do that? How can we compute attention with each GPU not being able to see the entire sequence for attention? Ring Attention!

---
## Ring Attention

While each GPU's **Query (Q)** remains stationary in its local memory, the **Keys (K)** and **Values (V)** are passed around the ring from one GPU to the next.

Say we have 4 GPUs and an input sequence of 4 tokens. Under context parallelism, the sequence is split evenly:

- **GPU 1** locally holds Token 1 ($Q_1, K_1, V_1$)
- **GPU 2** locally holds Token 2 ($Q_2, K_2, V_2$)
- **GPU 3** locally holds Token 3 ($Q_3, K_3, V_3$)
- **GPU 4** locally holds Token 4 ($Q_4, K_4, V_4$)

To compute the full attention, the system executes a loop that takes 4 time steps. During _every_ step, each GPU simultaneously performs three successive operations:

1. **Send (Asynchronously):** Send the K and V blocks it currently holds to the _next_ GPU in the ring. Because this is a "non-blocking" operation, the GPU initiates the transfer but doesn't sit idle waiting for it to finish.
    
2. **Compute (Locally):** While the network is busy sending data, the GPU calculates the attention scores for its stationary Q against the K and V blocks it _currently_ holds in memory. It computes a partial chunk of the standard attention formula:
    
    - $Softmax(\frac{QK^T}{\sqrt{d}}) * V$
    
3. **Receive & Update:** The GPU waits to receive the incoming K and V blocks from the _previous_ GPU in the ring. Once received, it circles back to Step 1, using these new blocks as its "current" keys and values.

To visualize how a single GPU sees the whole sequence, here is what is happening strictly inside **GPU 1** (which holds $Q_1$ permanently) across the 4 steps:

| **Time Step** | **K & V currently held** | **Computation on GPU 1**   | **Action at end of step**                                  |
| ------------- | ------------------------ | -------------------------- | ---------------------------------------------------------- |
| **Step 1**    | $K_1, V_1$               | Attends Token 1 to Token 1 | Sends $K_1, V_1$ to GPU 2. Receives $K_4, V_4$ from GPU 4. |
| **Step 2**    | $K_4, V_4$               | Attends Token 1 to Token 4 | Sends $K_4, V_4$ to GPU 2. Receives $K_3, V_3$ from GPU 4. |
| **Step 3**    | $K_3, V_3$               | Attends Token 1 to Token 3 | Sends $K_3, V_3$ to GPU 2. Receives $K_2, V_2$ from GPU 4. |
| **Step 4**    | $K_2, V_2$               | Attends Token 1 to Token 2 | Finished. (No send required).                              |
|               |                          |                            |                                                            |

![[ring-attention (1).gif|500]]

But this solution isn't perfect, why? Different GPUs get different workloads, take this example of 4 GPUs and 16 tokens, where each GPU receives 4 consecutive tokens:

- GPU 1 gets tokens 1–4
- GPU 2 gets tokens 5–8
- GPU 3 gets tokens 9–12
- GPU 4 gets tokens 13–16

In causal attention, each token can only attend to itself and previous tokens. That means the amount of valid attention work depends on how late in the sequence a token appears. Earlier tokens have fewer keys to attend to, while later tokens have many more.

So in this split:
- GPU 1 only needs to compute attention for tokens 1–4 against keys 1–4
- GPU 2 needs tokens 5–8 to attend over keys 1–8
- GPU 3 needs tokens 9–12 to attend over keys 1–12
- GPU 4 needs tokens 13–16 to attend over keys 1–16

This creates a triangular workload: later GPUs have more useful work than earlier ones, and in ring attention, this becomes a practical inefficiency. 

GPU 4 still needs the earlier key/value blocks 1–4, 5–8, and 9–12 to be passed around the ring so it can finish computing attention for tokens 13–16. But GPU 1 only needs its own local block 1–4. After that, it has no more useful causal attention work to do, so it can end up mostly idle while later GPUs are still computing.

![[HF_ULTRASCALE_PLAYBOOK 17.jpg|500]]

[[HF_ULTRASCALE_PLAYBOOK.pdf#page=103&rect=62,309,367,563|HF_ULTRASCALE_PLAYBOOK, p.103]]

So... what to do? Ring attention already gave inspiration for distributing K, Q, V values across different GPUs, but now we need to solve this even load problem between GPUs.

---
## Zig Zag Attention

What we can do instead is pair cheap early tokens with expensive late tokens so each GPU gets about the same total amount of causal work, as a detailed discussion in [here](https://github.com/zhuzilin/ring-flash-attention/issues/2#issuecomment-22367%2046166)
In the 16-token example:
- token 1 has 1 valid key
- token 16 has 16 valid keys
- together: 17

Then:
- token 2 has 2 valid keys
- token 15 has 15 valid keys
- together: 17

And so on.

So instead of assigning contiguous blocks like:
- GPU 1: 1–4
- GPU 2: 5–8
- GPU 3: 9–12
- GPU 4: 13–16

you could assign a balanced mixture like:
- GPU 1: 1, 2, 15, 16
- GPU 2: 3, 4, 13, 14
- GPU 3: 5, 6, 11, 12
- GPU 4: 7, 8, 9, 10
- 
That way, each GPU gets some cheap rows and some expensive rows, instead of one GPU getting only the hardest late-sequence rows.

![[HF_ULTRASCALE_PLAYBOOK 18.jpg|500]]

[[HF_ULTRASCALE_PLAYBOOK.pdf#page=106&rect=49,324,316,552|HF_ULTRASCALE_PLAYBOOK, p.106]]

And we could still use ring attention's idea to pass the K and V tensors around.