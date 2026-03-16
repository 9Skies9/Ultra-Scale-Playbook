Up to this point, you know what an LLM is, you've seen some techniques on how to improve their capabilities, how to evaluate their performance... but, we've been still toying around just on a single device to this point, and haven't touched much about how GPUs work, how to use lots of them at the same time for training, or inference.

Large labs do publish papers describing the ideas behind their state-of-the-art models, but the infrastructure to actually train them is rarely open. This book closes that gap.

We'll walk through the techniques behind training LLMs at scale in 3 steps:
1. basic theory
2. the code
3. real training benchmarks

Every technique in this book, no matter how sophisticated it looks is ultimately solving one of three problems: 
1. you're running out of memory
2. you're wasting compute due to data transfer
3. your GPUs are sitting idle waiting for communication overhead

In many places, we'll see that we can trade one of these (computation, communication, memory) off against another, finding the right balance is key to scaling training.
