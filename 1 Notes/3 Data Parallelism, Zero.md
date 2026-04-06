> [!PDF|yellow] [[HF_ULTRASCALE_PLAYBOOK.pdf#page=43&selection=3,0,9,1&color=yellow|HF_ULTRASCALE_PLAYBOOK, p.43]]
> > The idea behind data parallelism (DP) is to replicate the model on several GPUs (we call the replicas “model instances”) and run forward and backward passes on different micro-batches of data in parallel on each GPU — hence the name data parallelism.

In data parallelism, we are indeed ‘wasting’ memory in the sense that the full model is duplicated across GPUs instead of being split across them. Each GPU computes gradients on a different micro-batch, and then those gradients are averaged across all replicas so that the model weights stay identical after the update.

![[ddp.gif]]

This averaging is called 'all reduce', which would make the workflow something like this... but that raises the problem of overhead in communications, as GPUs have to wait around for the averaging operations to happen.
- the 0,1,2 here means layers in a neural network.

![[HF_ULTRASCALE_PLAYBOOK.jpg]]

[[HF_ULTRASCALE_PLAYBOOK.pdf#page=42&rect=28,273,334,351&color=yellow|HF_ULTRASCALE_PLAYBOOK, p.42]]

But, we can see the GLARING big overhead when the GPUs are just waiting around for the 'all reduce' to happen, what to do instead? 

While backward is still computing gradients layer by layer, the system can already start synchronizing and averaging the gradients for layers whose backward pass for calculating gradients are finished.

We are averaging each parameter tensor’s gradient with its matching copy on the other GPUs as soon as that tensor’s gradient is ready.

![[HF_ULTRASCALE_PLAYBOOK 2.jpg|500]]

[[HF_ULTRASCALE_PLAYBOOK.pdf#page=48&rect=31,473,337,564&color=yellow|HF_ULTRASCALE_PLAYBOOK, p.48]]

We can even combine this idea with what we learnt in chapter 2's gradient acclamation, why bother have the GPUs talk to each other at every backward pass for updating the parameters? Just talk to each other every x gradient accumulation steps and perform all reduce then.

This allows us to update our bath size equation, simply by multiplying by the number of GPUs (parallel instantness) which we have.

So the batch-size equation becomes

- $bs = gbs = mbs \times grad\_acc \times dp$

where:

- $mbs = \text{micro-batch size per GPU}$
- $grad\_acc = \text{number of gradient accumulation steps}$
- $dp = \text{number of GPUs (data-parallel replicas)}$


And this is where we get to a basic conception of why big companies buy like hundreds and thousands of GPUs for training these LLMs... you could make the batch size rack up based on however many GPUs you have.

> [!PDF|yellow] [[HF_ULTRASCALE_PLAYBOOK.pdf#page=55&selection=3,41,19,37&color=yellow|HF_ULTRASCALE_PLAYBOOK, p.55]]
> > Let’s say we want to train a recent model with a gbs of 4M tokens and a sequence length of 4k. Our batch size will thus be 1,024 samples (we pick the closest power of 2). Let’s assume we observe that a single GPU can only fit mbs=2 in memory, and we have 128 GPUs available for training. 
> > 
> > This means with 4 gradient accumulation steps, we’ll achieve our goal of 1,024 samples or 4M tokens per training step. Now, what if we suddenly have 512 GPUs available? We can achieve the same gbs by keeping mbs=2 and setting the number of gradient accumulation steps to 1, which will result in faster training!

You thought it was that easy! Nope, if we just add GPUs and more GPUs, overhead communication grows and grows, and throughput just drops to the ground.

![[HF_ULTRASCALE_PLAYBOOK 3.jpg|500]]

[[HF_ULTRASCALE_PLAYBOOK.pdf#page=56&rect=37,459,185,556&color=yellow|HF_ULTRASCALE_PLAYBOOK, p.56]]

And also... we haven't looked over the problem when we have models so large that they can't fit onto 1 GPU, just more and more problems! Who doesn't love that? 

Once naive data parallelism stops being enough, we need other strategies, the book looks at sharding here, with the implementation being ZeRO.

---
## Zero Redundancy Optimizer (ZeRO)

The simple idea behind ZeRO is simply dividing the optimizer states, gradients, and parameters across the GPUs.

This approach is organized into three possible optimization stages:
- ZeRO-1: optimizer state partitioning
- ZeRO-2: optimizer state + gradient partitioning
- ZeRO-3: optimizer state + gradient + parameter partitioning

Assume that our model has $\Psi$ parameters, then:

 $\underbrace{4\Psi}_{\text{parameters}} \;+\; \underbrace{4\Psi}_{\text{gradients}} \;+\; \underbrace{(4\Psi + 4\Psi)}_{\text{Adam optimizer state }(m,v)} \;=\; 16\Psi \text{ bytes}$

![[HF_ULTRASCALE_PLAYBOOK 4.jpg]]

[[HF_ULTRASCALE_PLAYBOOK.pdf#page=60&rect=37,389,326,556|HF_ULTRASCALE_PLAYBOOK, p.60]]

We'll start with ZeRO-1, where it only spits the optimizer states onto the different GPUs, each keeping only $1/{N_d}$ of the optimizer states. This means the forward pass + backward pass stays all the same, but the optimizing step changes. 

After backward, although each GPU has computed local gradients for the full model, it only needs the final reduced gradients for the parameter shard whose optimizer states it owns. 

Thus, a reduce-scatter is performed: the corresponding gradients are combined across GPUs (usually averaging), and each GPU receives only the reduced gradient shard matching its optimizer-state shard.

Then, each GPU performs a local optimizer step only on the parameter shard whose optimizer states it owns, updating only a subset of the parameters. Therefore, the updated parameter shards must then be shared so that every GPU can reconstruct the full parameter set for the next forward pass, an all-gather.

![[HF_ULTRASCALE_PLAYBOOK 6.jpg|500]]

[[HF_ULTRASCALE_PLAYBOOK.pdf#page=62&rect=30,246,337,561|HF_ULTRASCALE_PLAYBOOK, p.62]]

It's kinda funny how the basic training idea has not changed at all: still a forward pass, compute a loss, run a backward pass, and update the weights. 

In the simple one-GPU picture, that all feels clean and straightforward. The headache starts when we try to scale to much bigger models and batch sizes, because then that same simple process starts crashing into hardware limits, memory limits, and communication overhead.

But anyways, we can take the idea of ZeRO-1 up a notch with ZeRO-2:

> [!PDF|yellow] [[HF_ULTRASCALE_PLAYBOOK.pdf#page=65&selection=7,0,9,48&color=yellow|HF_ULTRASCALE_PLAYBOOK, p.65]]
> > Since on each replica we only need to have the gradient shard corresponding to its optimizer state shard, it makes sense to shard gradients as well, similarly to the optimizer states.

Still, the forward pass + backward pass stays all the same (as the entire model's gradients are calculated on each GPU). But as each GPU only needs the gradient shard corresponding to the optimizer-state shard it owns, the gradients are reduce-scattered across GPUs, so that each GPU receives and stores only its own reduced gradient shard.

![[HF_ULTRASCALE_PLAYBOOK 5.jpg|500]]

[[HF_ULTRASCALE_PLAYBOOK.pdf#page=64&rect=31,265,338,559|HF_ULTRASCALE_PLAYBOOK, p.64]]

But this means the GPU has to still store all the parameters at all time, so... can we also shard that? Yes, and that's ZeRO 3.

We basically extend our idea for the gradient shard-ing to parameters in the network. Where instead of every GPU storing a full copy of the parameters, each GPU stores only a shard of the parameters, along with the matching gradient shard and optimizer-state shard for that same portion of the model.
- Shard doesn't equal layer! A layer could be multiple shards if it's too large to fit into 1, a shard could be layer n + parts of layer n+1 if layer n is too small for 1 shard

Now you might wonder how forward/backward passes look like with all this shard-ing.
- an all-gather happens during the forward process, and all GPUs gather all the parameters from the other GPUs to perform the forward pass
- a reduce-scatter happens during the backward pass, where all GPUS calculate the full gradient vector, but each GPU only stores the gradient shard matching with it's parameters (through averaging over all other GPUs gradients for this shard)

You might go wow... this seems incredibly stupid for an idea, there's so much overhead going on with all the back-and-fourth transfer of data, right? Also, how does this actually help us save emory? Aren't we still loading all the parameters/gradients onto 1 GPU?

Now is the smart part, why load all parameters at the same time? Assume we have n layers in the network:
- Models go layer by layer, so just let all GPUs load layer 1 from the relevant shards, calculate the forward pass, discard layer 1 from memory of the GPUs, then let all GPUs load layer 2, calculate the forward pass... and so on
- This goes same for gradients as well, let all GPUs calculate layer n's gradient (then average), the resulting gradient is stored as shards on the GPUs that own the corresponding parameter shards. Then let all GPUs calculate layer n-1's gradient (then average), save that to the respective GPUs... and so on.

In this way, yes our GPUs will have some overhead in communication through all-gather and reduce-scatter, but those could happen while we are doing calculations:
1. gather parameters for layer 0
2. forward pass for layer 0
3. gather parameters for layer 1 simultaneously
4. forward pass for layer 1
5. gather parameters for layer 2 simultaneously
6. and onwards...

![[HF_ULTRASCALE_PLAYBOOK 7.jpg|500]]

[[HF_ULTRASCALE_PLAYBOOK.pdf#page=66&rect=31,159,337,563|HF_ULTRASCALE_PLAYBOOK, p.66]]

> [!PDF|yellow] [[HF_ULTRASCALE_PLAYBOOK.pdf#page=68&selection=3,0,27,2&color=yellow|HF_ULTRASCALE_PLAYBOOK, p.68]]
> > This may sound like a lot of communication overhead, but it’s actually not a big deal, as we can overlap the communication of the parameters for the next layer with the forward pass of the current layer in what is called prefetching. With prefetching, we all-gather the weights for Layer n + 1 while we do the forward pass for Layer n, and similarly, we all-gather the weights for Layer n-1 while doing the backward pass for Layer n. 

And we can see how through this shard-ing, the memory it takes on 1 GPU falls drastically.

![[HF_ULTRASCALE_PLAYBOOK 8.jpg|500]]

[[HF_ULTRASCALE_PLAYBOOK.pdf#page=69&rect=67,394,360,556|HF_ULTRASCALE_PLAYBOOK, p.69]]

- But what about activations? Even though re-computation and gradient accumulation can help, they do not fully solve the problem. Activation memory still grows with sequence length and batch size, and we do not want hardware limits to force us into only short contexts or tiny batches. 
- So if ZeRO handles model states, can we also do something about the computation itself? That's for the next chapter








