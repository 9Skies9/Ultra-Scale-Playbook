Still, what's the best combination? What's the best way to use these parallelism tools that we got? Well, it depends on your GPU case, how many GPUs do you have, the memory available per GPU, the transfer speed between your GPUs, etc.

The whole point is how should you combine parallelism methods so that:

1. the model fits in memory
2. training is efficient rather than being dominated by communication overhead


The team benchmarked several **thousand** distributed configurations to see which ones are the best empirically, and here's a funny note by them:

![[HF_ULTRASCALE_PLAYBOOK 27.jpg|500]]

[[HF_ULTRASCALE_PLAYBOOK.pdf#page=158&rect=29,277,336,339|HF_ULTRASCALE_PLAYBOOK, p.158]]

With a sequence length of 4,096 and a global batch size of 1M tokens, they played around configuring the combination of different parallelism strategies to different degrees with 1–64 nodes of 8xH100s.

![[Pasted image 20260330122329.png]]

> [!PDF|yellow] [[HF_ULTRASCALE_PLAYBOOK.pdf#page=159&selection=9,0,13,84&color=yellow|HF_ULTRASCALE_PLAYBOOK, p.159]]
> > For each combination, the configuration details include data parallelism (DP), tensor parallelism (TP), pipeline parallelism (PP), gradient accumulation steps (GAS), microbatch size (MBS), and ZeRO optimization stage. The color intensity indicates the model FLOPs utilization (MFU), with brighter colors representing higher efficiency.


We can see in general:
- more nodes = less throughput
- larger models = less throughput (because more nodes)
- implementation matters A LOT for any parallelism


On the website, they ran a LOT more experiments, you can click through them to see what worked and what didn't (the stuff in the center is better).
![[Screenshot 2026-03-30 at 12.33.35 PM.png]]

A crucial note they've mentioned is about how they assumed computation and communication could be overlapped between GPUs without impacting throughput, but... it isn't like that, we need to understand about GPUs better for how to optimize any of these parallelism tricks.

God, back to the depth of CUDA...


