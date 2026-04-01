> [!PDF|yellow] [[HF_ULTRASCALE_PLAYBOOK.pdf#page=142&selection=3,0,57,21&color=yellow|HF_ULTRASCALE_PLAYBOOK, p.142]]
> > Congratulations, reader! You have now seen all five parallelism strategies you can use to scale model training:
> > 
> > 1. Data parallelism (DP) — along the batch dimension 
> > 2. Tensor parallelism (TP) — along the hidden dimension 
> > 3. Sequence and context parallelism (SP / CP) — along the sequence dimension 
> > 4. Pipeline parallelism (PP) — along the model layers 
> > 5. Expert parallelism (EP) — along the model experts 
> > 
> >  as well as the three ZeRO strategies that can be combined with data parallelism for memory reduction: 
> >  
> > 1. ZeRO-1 — sharding optimizer states among the DP replicas 
> > 2. ZeRO-2—sharding optimizer states and gradients among the DP replicas 
> > 3. ZeRO-3 — sharding optimizer states, gradients, and parameters among the DP replicas

The mental model is that all of these parallelism strategies work in different dimensions, and we need to honestly ask ourselves: “which axes do I need to split to make this particular model fit and run efficiently?”

A very blunt summary:
- $DP$ is the default
- ZeRO modifies $DP$
- $TP$ is for wide layers
- $PP$ is for deep models
- $SP/CP$ are for long sequences
- $EP$ is for MoE

But... it's like suddenly having to juggle 8 new balls altogether, and having no idea how they interact with each other. We can compare them, and see which ones work together, which ones conflict one another.

![[HF_ULTRASCALE_PLAYBOOK 26.jpg|500]]

[[HF_ULTRASCALE_PLAYBOOK.pdf#page=152&rect=37,241,327,519&color=yellow|HF_ULTRASCALE_PLAYBOOK, p.152]]

## Zero 3 and PP

Starting with Zero (3) and PP, they both aim to reduce the memory load of model parameters on a GPU

- Pipeline parallelism partitions the model by depth.
- ZeRO-3 partitions model states across replicas.

Pipeline parallelism assigns different contiguous layer blocks to different GPUs, so one GPU may own layers 1–4, another 5–8, and so on. Each GPU stores full parameters for its own layers, and the main thing passed between GPUs is activations.

ZeRO-3 shards parameters (and gradients, and optimizer states, but that's another story), so each GPU holds only a fraction of a layer’s states at a time. When a layer is needed, its parameters are gathered, used for computation, and then released or reshaped as needed.

![[HF_ULTRASCALE_PLAYBOOK 24.jpg|500]]

[[HF_ULTRASCALE_PLAYBOOK.pdf#page=143&rect=70,372,357,519&color=yellow|HF_ULTRASCALE_PLAYBOOK, p.143]]

Tensor parallelism and sequence parallelism are usually paired to make large intra-layer computation feasible. Tensor parallelism splits the hidden-dimension computation of large layers across GPUs, while sequence parallelism reduces the activation-memory overhead for token-independent operations such as LayerNorm and dropout.

![[Pasted image 20260328171758.png|700]]

Context parallelism targets the long-sequence problem by sharding activations along the sequence dimension across GPUs. It is especially useful for attention, since attention memory becomes prohibitive at very large context lengths.

Expert parallelism is for Mixture-of-Experts models. Instead of every GPU storing and computing every expert MLP, the experts are distributed across GPUs. Since each token is routed to only a small subset of experts, each GPU only handles the experts assigned to it.

![[Pasted image 20260328172216.png|700]]


They are all aimed for different purposes! And that's the fun part of all of this!

![[HF_ULTRASCALE_PLAYBOOK 25.jpg|500]]

[[HF_ULTRASCALE_PLAYBOOK.pdf#page=148&rect=37,113,325,272&color=yellow|HF_ULTRASCALE_PLAYBOOK, p.148]]

---

## Summarizing It All!

Yes! We can place all these techniques that we've discussed into a single diagram! And after all this learning, get a overview of the memory saving techniques for all these strategies.

![[Pasted image 20260328173037.png]]

And how these different techniques when put together, actually reduces memory load in our GPUs.

![[Pasted image 20260328173229.png|700]]

So, that's the parallelism strategies for pre-training LLMs, but none is universally best as a silver bullet. Each one improves scaling along a particular dimension while paying a cost in something else. The next chapter turns to the practical question of which combinations are most effective in real training setups.