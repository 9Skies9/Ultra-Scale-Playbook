> [!PDF|yellow] [[HF_ULTRASCALE_PLAYBOOK.pdf#page=111&selection=27,1,35,63&color=yellow|HF_ULTRASCALE_PLAYBOOK, p.111]]
> > ipeline parallelism is a simple but powerful technique — we split our model’s layers across multiple GPUs! For example, if we have 8 GPUs, we could put layers 1–4 on GPU 1, layers 5–8 on GPU 2, and so on. This way, each GPU only needs to store and process a portion of the model’s layers, significantly reducing the memory requirements per GPU.

But this has a huge problem, GPU 2 has to wait for GPU 1 to finish computing layers 1-4 before it can start computing layers 5-8 using GPU 1's outputs, this... is like a bane of existence in parallelism, why would you do that?

![[120 CS/123 AI/3 NLP/4 Ultra Scale Playbook/1 Notes/attachments/HF_ULTRASCALE_PLAYBOOK 19.jpg|500]]

[[HF_ULTRASCALE_PLAYBOOK.pdf#page=114&rect=37,474,327,556&color=yellow|HF_ULTRASCALE_PLAYBOOK, p.114]]

Well... it's just how this technique works! Naively, it's terrible due to this incredibly low throughput, but we can fight  some of the inefficiencies of pipeline stages by adding more micro-batches.

This approach is called the all forward, all backward schedule. 

![[120 CS/123 AI/3 NLP/4 Ultra Scale Playbook/1 Notes/attachments/HF_ULTRASCALE_PLAYBOOK 20.jpg]]

[[HF_ULTRASCALE_PLAYBOOK.pdf#page=114&rect=38,360,326,439&color=yellow|HF_ULTRASCALE_PLAYBOOK, p.114]]

There is another one called 1 forward, 1 backward schedule, but... eh, I don't want to talk about it, as it had barely a performance increase.

The idea is simply, instead of performing all forward passes then the backward passes, it starts performing backward passes whenever a micro batch's forward pass is completed.

![[120 CS/123 AI/3 NLP/4 Ultra Scale Playbook/1 Notes/attachments/HF_ULTRASCALE_PLAYBOOK 21.jpg]]

[[HF_ULTRASCALE_PLAYBOOK.pdf#page=118&rect=37,474,327,556&color=yellow|HF_ULTRASCALE_PLAYBOOK, p.118]]

Though, it does save some memory, as we only need to store activations for p micro-batches instead of m micro-batches
- p: number of pipeline parallelism
- m: number of micro-batches

We could go crazier than this through interleaving, where GPU 1 originally holds layers 1-4, and GPU 2 holds layers 5-8, we could make GPU 1 hold layers 1-2 and 9-10, GPU 2 hold layers 3-4 and 11-12.

So with 16 layers and 4 GPUs, a simple interleaved setup might look like:

- GPU 1: layers 1–2 and 9–10
- GPU 2: layers 3–4 and 11–12
- GPU 3: layers 5–6 and 13–14
- GPU 4: layers 7–8 and 15–16

This... create some crazy scheduling, it reduce pipeline bubbles and idle time, but it comes with more overhead and more complexity.

![[120 CS/123 AI/3 NLP/4 Ultra Scale Playbook/1 Notes/attachments/HF_ULTRASCALE_PLAYBOOK 22.jpg|500]]

[[HF_ULTRASCALE_PLAYBOOK.pdf#page=124&rect=37,470,327,556&color=yellow|HF_ULTRASCALE_PLAYBOOK, p.124]]

But even that's not enough...

> [!PDF|yellow] [[HF_ULTRASCALE_PLAYBOOK.pdf#page=129&selection=7,0,12,5&color=yellow|HF_ULTRASCALE_PLAYBOOK, p.129]]
> > Even more sophisticated ways to reduce the bubble have recently been proposed that reach close to a “zero bubble” regime, such as the pipeline implementation approach in DeepSeek-V3/R1, called DualPipe.

(Bruh I'm not explaining this one, help me)