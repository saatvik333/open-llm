# Part II — Neural Networks Learn Language

Before the transformer, we need its components: differentiable functions, learned representations, sequence objectives, and tokens. This part builds a language model from small pieces.

# Neural Networks Without Mysticism

A neural network is a parameterized function assembled from simple operations. “Neural” is historical inspiration, not a faithful simulation of a biological brain. Training adjusts parameters so that outputs score well under an objective.

## One artificial neuron

Given input vector **x**, weight vector **w**, and bias `b`, a neuron computes:

`z = x · w + b`

then applies a nonlinear activation:

`h = φ(z)`

A perceptron uses a hard threshold. Modern networks use differentiable or almost-everywhere differentiable activations:

- **sigmoid:** maps to `(0, 1)`, useful for gates or binary probabilities but saturates;
- **tanh:** maps to `(-1, 1)`;
- **ReLU:** `max(0, z)`, simple and effective;
- **GELU:** smoothly gates by input value and became common in transformers;
- **SiLU/Swish:** `z × sigmoid(z)`, often used in gated feed-forward blocks.

Without the activation, many layers collapse into one linear map. Nonlinearity lets the network form curved decision boundaries and conditional computations.

## A multilayer perceptron

A two-layer multilayer perceptron (MLP) is:

`h = φ(xW₁ + b₁)`

`y = hW₂ + b₂`

The first matrix expands or changes the feature space; the activation selects a nonlinear pattern; the second combines features into outputs. Transformer blocks contain a position-wise MLP applied independently to each token after attention has mixed information across positions.

A common gated variant resembles:

`MLP(x) = (SiLU(xW_gate) ⊙ xW_up)W_down`

where `⊙` is elementwise multiplication. The gate lets one learned projection control another. With a SiLU gate this form is called SwiGLU; the paper that tested such gated variants in transformer feed-forward layers found quality gains over ReLU and GELU in its experiments [R96].

## Universal approximation—and its limits

Under technical conditions, a sufficiently wide neural network can approximate any continuous function on a compact region. This theorem says a representation exists. It does not say training will find it, that data identify it, that the network is efficient, or that it generalizes. Expressivity is only one piece.

## Loss, forward pass, backward pass

A training step has four conceptual stages:

1. **Forward:** compute predictions from inputs.
2. **Loss:** compare predictions with targets.
3. **Backward:** compute gradients of loss with respect to parameters.
4. **Update:** use an optimizer to change parameters.

Pseudocode:

```python
for batch in data:
    prediction = model(batch.inputs)
    loss = objective(prediction, batch.targets)
    gradients = backward(loss, model.parameters)
    optimizer.update(model.parameters, gradients)
```

Framework syntax hides device transfers, mixed precision, distributed synchronization, gradient scaling, accumulation, and failure recovery, but the logical loop remains.

## Initialization

If every neuron starts with identical weights, symmetry gives them identical gradients and they remain copies. Random initialization breaks symmetry. Its scale must preserve useful activation and gradient variance through depth. Xavier/Glorot and He initialization derive scales for particular activation assumptions.

Transformers add residual pathways and normalization, so initialization may be adjusted with depth. At large scale, a seemingly small choice can decide whether a costly run is stable.

## Residual connections

A residual layer computes:

`y = x + F(x)`

Instead of replacing **x**, the block learns a correction. The identity path gives information and gradients a direct route through many layers. Shapes must match, or a projection adapts them.

Residual streams are a useful transformer mental model: each block reads the current shared representation and writes an update. This is an abstraction, not proof that each layer has a simple isolated role.

## Normalization

Normalization controls activation scale. Layer normalization standardizes features within each token and then applies learned scale and shift. RMS normalization uses root-mean-square magnitude without subtracting the mean and is common in current LLMs.

A schematic RMSNorm is:

`RMSNorm(x) = g ⊙ x / sqrt(mean(x²) + ε)`

where **g** is learned. Normalization improves optimization but also changes representation geometry and precision behavior.

**Pre-norm** transformer blocks normalize before attention or the MLP; **post-norm** blocks normalize after adding a residual. Pre-norm often stabilizes deep training. Variants continue to evolve. One fully open model family moved normalization to the *outputs* of the attention and MLP sublayers, inside the residual branch, a placement its authors attribute to earlier work, and combined it with other stability measures described in the next section [R97]. Where the norm sits changes both gradient flow and the scale of the residual stream, so it is part of the recipe, not a cosmetic choice.

## Dropout and train/evaluation modes

Dropout randomly zeros activations during training and rescales survivors. At evaluation it is disabled. Other layers may also behave differently between modes. Accidentally evaluating in training mode or training in evaluation mode causes subtle bugs.

Many large language pretraining recipes use little or no dropout when data are abundant, while smaller fine-tunes may benefit. Regularization is contextual.

## Keeping large training runs stable

A large pretraining run can fail in ways a small run never shows. The loss may jump upward for a few steps and recover (a **loss spike**), or it may diverge and never come back. The largest PaLM model spiked about 20 times despite gradient clipping. Its team’s remedy was operational: restart from a checkpoint about 100 steps before the spike and skip roughly 200 to 500 batches. The spike did not recur at the same point, and the team wrote that it had not found a principled fix [R85]. Since then, researchers have linked several instabilities to specific quantities and added small, targeted patches. This section covers the patches now common in published recipes, and labels what kind of evidence each rests on.

**Attention logits that grow without bound.** Part III explains attention in detail. For now, you need only know that attention scores are dot products of query and key vectors, passed through softmax. Nothing in the architecture stops those dot products from growing during training. When they become huge, softmax outputs become almost one-hot: attention locks onto one position, and gradients for the other positions shrink toward zero. *Empirical finding:* while scaling a vision transformer to 22 billion parameters, researchers saw the loss diverge after a few thousand steps in models of about 8 billion parameters. The cause was extremely large attention logits, which produced attention weights with near-zero entropy. Applying layer normalization to the queries and keys before the dot product prevented the divergence [R98]. The same idea, called **QK-normalization**, had appeared in 2020 as an l2 normalization with a learned scale, proposed for low-resource translation [R99].

The sketch below shows the effect. It scales queries and keys of dimension 64 by 1, 4, and 16, then compares a plain scaled dot product with one that first rescales every vector to root-mean-square 1. Real QK-normalization also learns a gain, which this sketch fixes at 1.

```python
import math
import numpy as np

rng = np.random.default_rng(0)
d, n = 64, 128
q0, K0 = rng.standard_normal(d), rng.standard_normal((n, d))

def attention(q, K):                       # largest logit and attention entropy (nats)
    s = K @ q / math.sqrt(d)
    p = np.exp(s - s.max())
    p /= p.sum()
    return round(float(s.max()), 1), round(float(-(p * np.log(p + 1e-30)).sum()), 2)

def unit_rms(x):                           # rescale each vector to root-mean-square 1
    return x / np.linalg.norm(x, axis=-1, keepdims=True) * math.sqrt(d)

for scale in (1, 4, 16):
    q, K = q0 * scale, K0 * scale
    print(scale, attention(q, K), attention(unit_rms(q), unit_rms(K)))
# 1 (1.9, 4.5) (2.1, 4.44)
# 4 (30.9, 0.32) (2.1, 4.44)
# 16 (494.6, 0.0) (2.1, 4.44)
```

Attention spread uniformly over 128 keys would have an entropy of ln 128 ≈ 4.85 nats. At scale 16 the plain version has a largest logit near 495 and an entropy of 0.00: a single winner. The normalized version is unchanged by scale. In a real network the vectors are learned, so the scale is not chosen by hand. QK-normalization removes growth in the length of the query and key vectors as a route to huge logits. The learned gain that remains can still grow, but it is a much smaller set of numbers to control.

**Output logits that drift (definition and empirical finding).** Softmax ignores a constant added to every logit, as the softmax section later in this part shows, so nothing in the loss prevents the logits from drifting to large values. Large numbers are stored coarsely: in a 16-bit format with 7 mantissa bits, values between 32 and 64 are 0.25 apart. PaLM added an auxiliary **z-loss**, `10⁻⁴ × (log Z)²`, where `Z = Σ exp(zᵢ)` is the softmax denominator. It pushes `log Z` toward 0, and the team reported that it improved training stability [R85]. A small-model study showed that two instabilities, attention-logit growth and drift of the output logits away from the log-probabilities, can be reproduced at small scale by training at high learning rates, and that the same mitigations work there [R100]. This makes cheap tests of a recipe possible.

Worked example: logits `[2, 1, 0]` have `Z ≈ 11.11` and `log Z ≈ 2.41`, so the penalty is `10⁻⁴ × 2.41² ≈ 0.0006`. Add 30 to every logit. The probabilities stay `[0.665, 0.245, 0.090]`, but `log Z` becomes 32.41 and the penalty becomes about 0.105, roughly 180 times larger. The term does not change which tokens are likely. It only removes the freedom to drift.

**A bundle of measures (reports from specific runs).** OLMo 2, a model family released with its data, code, and logs, documented the changes it made after an earlier run in the same family showed frequent loss spikes preceded by growing gradient norms. They were QK-normalization with RMS normalization; a z-loss; normalizing sublayer outputs, as described above; initializing every parameter from a normal distribution with mean 0 and standard deviation 0.02; excluding embeddings from weight decay; lowering the AdamW epsilon from 10⁻⁵ to 10⁻⁸; and filtering training documents that contain long repeated n-gram sequences, which the authors found in many batches that preceded spikes [R97]. Gemma 3 replaced the soft-capping used in Gemma 2, which squashes attention scores into a fixed range, with QK-normalization [R101]. Kimi K2 pairs the Muon optimizer (Part I) with multi-head latent attention, a design that compresses keys and values (Part III, “Multi-head latent attention”). Its authors write that QK-normalization is not applicable there, because the key matrices are never fully materialized at inference. They use **QK-Clip** instead: whenever a head’s largest logit in a batch exceeds a threshold, the query and key projection weights are rescaled after the update to bring it back. They report pretraining on 15.5 trillion tokens without a single loss spike [R93].

::: warning Patches are not proofs
Each measure above has a cost. QK-normalization adds operations and changes the geometry attention can learn. A z-loss adds a hyperparameter. Clipping weights treats a symptom. Skipping batches hides a cause. The reports cited here describe specific recipes at specific scales; none proves that a fix generalizes. Ablate at small scale, then watch the quantities that move first: the largest attention logit per layer, the global gradient norm, and the size of the output logits. Narrow number formats, covered in Part I, leave less headroom for all three.
:::

# Representations and Embeddings

A representation turns an object into features that make useful transformations easier. Neural networks learn representations jointly with their objective.

## From one-hot IDs to dense vectors

Suppose a vocabulary has 50,000 tokens. A one-hot vector for token 17 has length 50,000, with one 1 and all other entries 0. It expresses identity but no similarity.

An embedding matrix **E** of shape `[vocabulary, d_model]` stores a dense learned vector for each token. Looking up token ID `i` selects row **Eᵢ**. This is mathematically equivalent to multiplying a one-hot vector by **E**, but lookup is efficient.

During training, gradients adjust embeddings so they support prediction. Tokens appearing in interchangeable contexts may develop related vectors. The exact geometry depends on corpus, architecture, objective, and later layers.

## Static versus contextual embeddings

Word2vec assigns a fixed vector to each vocabulary item. “Bank” has one vector whether it refers to a river or finance. A transformer begins with a token embedding but repeatedly mixes context, producing a different hidden state for each occurrence.

A contextual representation can encode syntax, sense, role, discourse, formatting, or task information. No single coordinate need correspond cleanly to one concept.

## Similarity and retrieval

Embedding models map a query and candidate passages into vectors. Retrieval ranks candidates by dot product, cosine similarity, or another learned score. This works because contrastive training brings related pairs closer and pushes unrelated pairs apart.

Similarity is task-relative. “Dog” is similar to “cat” taxonomically, to “bark” associatively, and to “wolf” evolutionarily. An embedding trained for web search may differ from one trained for duplicate-question detection.

## How embedding models are trained

The previous section said that contrastive training pulls related pairs together. Here is the mechanism.

**Contrastive loss (definition).** Gather pairs that should match: a question and a passage that answers it, two paraphrases, an image and its caption. In a mini-batch of `B` pairs, take one query. Its own partner is the **positive**. The other `B − 1` partners in the batch serve as free **negatives**, called in-batch negatives [R38, R41]. Score the query against every candidate with cosine similarity, divide each score by a **temperature** `τ`, and apply softmax. The loss is the negative log of the probability given to the positive:

`loss = −log( exp(s⁺/τ) / Σⱼ exp(sⱼ/τ) )`

where `s⁺` is the positive’s similarity and the sum runs over the positive and all negatives. This is the cross-entropy of earlier chapters with candidates in place of vocabulary tokens. Contrastive predictive coding named this form of loss InfoNCE [R102].

**Worked example (illustrative numbers, computed).** Let the cosine similarities be 0.82 to the positive and 0.41 and 0.35 to two negatives, with `τ = 0.1`. Dividing gives 8.2, 4.1, and 3.5. Softmax gives the positive a probability of 0.975, so the loss is 0.025. Now suppose the first negative is a near-duplicate scoring 0.80. The positive’s probability falls to 0.547 and the loss rises to 0.603. With `τ = 1` and the original easy negatives, the loss is 0.828, because the scores barely separate. Two lessons follow. Easy negatives teach little once the model separates them, so training pipelines mine **hard negatives**, such as passages that share many words with the question but do not answer it [R38]. And temperature sets how sharply the loss punishes close competitors.

**Language-model backbones (empirical trend).** Embedding models are now often initialized from decoder-only language models rather than from encoder-only ones. One recipe fine-tuned open decoder-only models with a standard contrastive loss on synthetic training data generated by proprietary language models, in fewer than 1,000 steps [R103]. The Qwen3 Embedding series, released in June 2025, starts from language models with causal attention, appends an end-of-sequence token, and takes the last layer’s hidden state at that token as the embedding. It concatenates an instruction with the input, so one model can embed for different tasks, and trains first on large-scale unsupervised data and then on supervised data [R104]. Benchmarks such as MTEB, which in its original release spanned 8 task types, 58 datasets, and 112 languages, test whether one model works across tasks [R105]. Public embedding benchmarks carry the same overfitting risk as any public benchmark (Part I, “Train, validation, and test”).

**Matryoshka embeddings (definition).** A standard embedding has a fixed length, say 1,024 numbers, and cutting it short ruins it. *Matryoshka representation learning* trains the model so that the first `m` numbers, for several sizes `m` such as 64, 128, 256, 512, and 1,024, each work as an embedding on their own. It adds a loss term for every prefix size [R106]. The authors report that the short prefixes were at least as accurate as separately trained low-dimensional models [R106]. In practice you can search with a short prefix, then rescore the best candidates with the full vector. Cutting a vector changes its length, so renormalize before using cosine similarity. Some current embedding models expose this as a custom-dimension option [R104].

Storage example (assumptions: 100 million documents, float32 values, before any index overhead): 1,024 dimensions need `100,000,000 × 1,024 × 4 bytes = 409.6 GB`. Keep 256 dimensions at one byte each, and the total is `100,000,000 × 256 × 1 byte = 25.6 GB`, 16 times smaller. What the shorter vectors cost in retrieval quality must be measured on your own data.

**One vector may not be enough.** A single vector per passage compresses everything about it. *Late interaction* keeps one vector per token. ColBERT encodes the query and the document separately with a language model, then scores a pair by taking, for each query token, its highest similarity to any document token, and summing over the query tokens [R107]. This stores far more numbers per document but matches finer detail. Single vectors also have a theoretical limit: for a given dimension, only so many different subsets of documents can be the top results for some query. Researchers showed that this limit appears even with free embeddings optimized directly on the test set, and built a deliberately simple dataset on which strong embedding models failed [R108]. Treat a single-vector embedding as a fast first stage, and combine it with lexical search, multi-vector scoring, or a reranker when coverage matters (Part V).

## Positional information

Token embeddings alone do not tell attention whether a word came first or last. Models add or incorporate position information.

**Learned absolute embeddings** assign a vector to each position. **Sinusoidal embeddings** use fixed waves at multiple frequencies. **Relative position biases** modify attention based on distance. **Rotary position embeddings (RoPE)** rotate query and key features by position-dependent angles so their dot products encode relative offsets [R20]. **ALiBi** adds head-specific linear distance biases [R21].

Length extrapolation remains delicate. A model trained mostly on short sequences may not use much longer context reliably even if the software accepts it. Position scaling methods change geometry but cannot create missing long-range training experience.

## Representation is not a database row

Facts are not generally stored as clean key–value records at one address. Parametric knowledge is distributed across weights and activated by context. Updating one fact can affect others, and exact provenance may be unavailable.

External retrieval is preferable when information must be fresh, attributable, revocable, access-controlled, or updated independently. Fine-tuning is better suited to behavior, style, format, domain adaptation, and repeated patterns—though the boundary is not absolute.

# From Counts to Neural Language Models

Language modeling estimates a distribution over sequences. The methods changed; the central probabilistic object did not.

## N-grams

An n-gram model assumes the next token depends only on the previous `n-1` tokens:

`P(x_t | x₁…x_{t-1}) ≈ P(x_t | x_{t-n+1}…x_{t-1})`

Counts estimate probabilities. Smoothing reserves probability for unseen sequences. Backoff uses shorter contexts when a long context is rare.

N-grams are fast, interpretable, and still useful in constrained systems. Their weakness is sparsity and rigid context identity: “the cat sat” and “a kitten rested” share meaning but no exact n-grams.

## Feed-forward neural language models

A neural model embeds context tokens and passes them through an MLP to predict the next token. Similar embeddings share statistical strength. But a fixed context window still limits dependence, and concatenating many embeddings becomes cumbersome.

## Recurrent neural networks

An RNN updates a hidden state:

`h_t = φ(x_t W_x + h_{t-1} W_h + b)`

`P(x_{t+1} | x_{≤t}) = softmax(h_t W_o)`

The state summarizes the prefix. Weight sharing handles variable lengths. In principle, information can travel indefinitely; in practice, gradients and finite state make long dependencies difficult.

LSTM and GRU architectures use gates to manage writing, retaining, and reading state. They were major advances in language, speech, and sequence modeling and remain useful for streaming or compact tasks.

## Encoder–decoder and attention

A sequence-to-sequence encoder reads an input; a decoder generates output. Compressing the whole input into one vector creates a bottleneck. Attention lets each decoder step form a weighted combination of encoder states.

This introduced a general pattern:

1. compare a query with a set of keys;
2. normalize match scores into weights;
3. combine corresponding values.

The transformer applies this pattern within a sequence and across many positions in parallel.

## Encoder-only, decoder-only, encoder–decoder

- **Encoder-only** models can attend bidirectionally across observed input. They excel at representation, classification, extraction, and masked-token objectives.
- **Decoder-only** models use causal masking so each position sees only its prefix. They naturally generate open-ended sequences.
- **Encoder–decoder** models encode an input bidirectionally and decode an output causally with cross-attention. They fit translation, summarization, and conditional generation.

Architecture is not destiny. Training scale, data, objectives, tools, and product design can matter more than the category name. The categories also blur. Embedding models are often built by adapting decoder-only language models (see “How embedding models are trained”), and masked-diffusion language models use a bidirectional transformer for generation (see “Beyond one next token at a time”).

# Tokenization: The Interface Between Text and Tensors

Models do not receive words directly. A tokenizer converts input into a sequence of integer IDs; a detokenizer reconstructs text from generated IDs.

## Why not one token per word?

A word vocabulary faces unknown words, spelling variants, rich morphology, URLs, source code, and many writing systems. A character vocabulary handles anything but creates long sequences and must learn common chunks from scratch. Subword tokenization compromises: common strings become single tokens; rare strings split into pieces.

Byte-level methods can represent arbitrary input without an unknown token. They may still group bytes into learned subwords for efficiency. A common design keeps a learned vocabulary but falls back to one token per UTF-8 byte for characters it has no entry for, so nothing is lost. PaLM’s tokenizer, with a vocabulary of about 256 thousand entries, worked this way [R85].

## Byte-pair encoding

A simplified byte-pair encoding (BPE) procedure begins with small units, counts adjacent pairs, and repeatedly merges the most frequent pair. After training, the ordered merge rules tokenize new text [R17].

Toy corpus:

```text
low lower newest widest
```

Starting with characters, frequent pairs such as `l+o`, `lo+w`, or `e+s` may merge. Real tokenizers handle whitespace markers, bytes, normalization, special tokens, and deterministic priority rules.

**Unigram language-model tokenization** begins with many candidate pieces and removes those whose absence least harms a probabilistic segmentation objective. SentencePiece popularized language-independent training directly from raw text [R18].

**Superword tokens.** Nearly all tokenizers share one assumption: a token never crosses a word boundary. Multi-word units such as “by the way,” languages written without spaces, and compounds that one language writes as several words and another as one all strain it. SuperBPE first learns ordinary subword merges, then lifts the whitespace restriction so that later merges can create “superword” tokens. With the vocabulary fixed at 200,000 entries it encoded a fixed text with up to 33 percent fewer tokens than standard BPE on average. In 8-billion-parameter models trained with the same size, vocabulary size, and training compute, the authors report better average accuracy over 30 downstream tasks and about 27 percent less inference compute [R109]. That is one group’s controlled comparison, not a settled default.

## Tokenization changes computation

Compute and billing often scale with tokens, not words. A tokenizer that encodes one language inefficiently gives it:

- shorter effective context in meaning per token;
- higher latency and cost;
- fewer long-range examples during training for a fixed token budget;
- more generation steps and more chances for error.

Code, numbers, and whitespace can also split unexpectedly. Never estimate token count from character count alone when a limit matters.

Three measured effects show how unequal the costs can be:

- **Language.** The same text translated into different languages can differ by up to 15 times in token count, and the gaps persist for tokenizers trained with multilingual support [R110]. Cost, latency, and how much material fits in context therefore differ by language.
- **Numbers.** Tokenizers handle digits differently. PaLM always splits numbers into single digits [R85]. GPT-3.5 and GPT-4 have separate tokens for groups of one, two, and three digits. A study of arithmetic in those models found that grouping digits from the right, forced by writing numbers with commas, improved accuracy compared with grouping from the left, and that the gap shrank as models scaled [R111].
- **Measurement.** Per-token loss is not comparable across tokenizers. A tokenizer-independent measure is **bits per byte**: the total loss converted to bits, divided by the number of UTF-8 bytes in the text [R84]. Hypothetical example: tokenizer A averages 4.5 bytes per token and has a loss of 2.00 nats per token. Tokenizer B averages 3.5 bytes per token and has 1.60 nats per token. B looks better per token, with perplexity 4.95 against 7.39, but bits per byte are `2.00 ÷ (ln 2 × 4.5) = 0.641` for A and `1.60 ÷ (ln 2 × 3.5) = 0.660` for B. A compresses the text better.

## Special tokens and chat templates

Models reserve tokens for boundaries or control, such as beginning/end of sequence, padding, masked positions, roles, tool calls, or modality placeholders. A chat template serializes messages into the exact sequence expected during post-training.

A conceptual template might be:

```text
<begin><system>You are concise.</system>
<user>Explain entropy.</user>
<assistant>
```

Using the wrong template can sharply degrade behavior. Role labels visible in an API are eventually converted to tokens, embeddings, masks, or control structures.

## Normalization and security

Unicode permits visually similar strings with different code points. Tokenizers may normalize some forms and preserve others. This affects accents, identifiers, emojis, and adversarial text. Invisible characters, bidirectional controls, homoglyphs, and unusual whitespace can bypass naive filters.

Security checks should operate at appropriate layers: raw bytes, normalized text, parsed structure, token sequence, and tool arguments. No single regex is a sufficient prompt-injection defense.

## Vocabulary size trade-offs

A larger vocabulary makes sequences shorter but enlarges the embedding and output matrices and gives rare entries fewer updates. A smaller vocabulary improves sharing but lengthens sequences. Multilingual coverage, modality tokens, code, and serving hardware all affect the choice.

**Arithmetic of size.** An embedding matrix has `vocabulary × d_model` entries. With 262,144 entries and an illustrative `d_model` of 4,096, that is about 1.07 billion parameters, and the output matrix doubles it unless the two are tied.

**Observed choices.** Vocabularies grew from tens of thousands of entries to over a hundred thousand. Llama 3 uses 128,000 entries, 100,000 from an existing tokenizer plus 28,000 added for non-English text, and reported that English compression rose from 3.17 to 3.94 characters per token relative to its predecessor’s tokenizer [R112]. Gemma 3 uses a tokenizer with 262 thousand entries, described as more balanced across non-English languages [R101]. OLMo 2 adopted a larger vocabulary than the earlier members of its family, citing a study of vocabulary scaling [R97].

**Empirical finding.** That study trained models from 33 million to 3 billion parameters and found that the compute-optimal vocabulary size grows with the compute budget. It predicted that a 70-billion-parameter model from 2023 would have been best served by at least 216,000 entries, seven times its 32,000 [R113]. This is a fitted prediction, not a law.

**Failure mode.** A tokenizer is usually built before, and separately from, training. Entries that exist in the vocabulary but almost never appear in the training data can become **under-trained tokens**, whose embeddings were barely updated and which can provoke odd behavior when they appear in a prompt. Researchers have developed ways to detect them from tokenizer analysis, model-weight indicators, and prompting, and found them in many models [R114].

## Beyond fixed tokenizers: bytes, patches, and dynamic chunks

Tokens are an engineering interface, not a natural law of language. Two research designs try to remove the fixed tokenizer.

**Patches grouped by predictability.** The Byte Latent Transformer (BLT) reads raw bytes but does not give every byte equal compute. A small byte-level language model estimates how uncertain the next byte is. High-uncertainty points start a new patch, so predictable stretches become long patches and hard stretches become short ones. A large transformer operates on patches, and small byte-level networks convert between bytes and patches [R115]. The authors report a controlled study up to 8 billion parameters and 4 trillion training bytes in which this approach matched tokenization-based models while improving inference efficiency and robustness [R115].

**Chunking learned end to end.** H-Net learns where to chunk jointly with the rest of the network. It replaces the pipeline of tokenizer, language model, and detokenizer with one model: early layers read bytes, a chunking step compresses them into fewer vectors, a main network processes those, and later layers expand them again. The structure can be nested. In a comparison matched for compute and data, a one-stage byte-level H-Net outperformed a strong transformer over BPE tokens, with larger gains on Chinese, code, and DNA, where tokenization heuristics are weaker [R116]. The paper appeared at ICLR 2026. On 28 October 2026 Cartesia Research, which describes H-Nets as its own 2025 work, published a blog post reporting efficiency gains for a further improved version; it was not peer reviewed and the architecture had not been released [R117].

**Trade-offs.** Bytes make sequences longer. English text averages roughly four bytes per token under a 128,000-entry tokenizer (derived from the 3.94 characters per token above, assuming about one byte per character), so a byte-level model must either process about four times as many positions or compress them again, which is what patches and chunks do. Removing the tokenizer also removes some of its failure modes, such as under-trained tokens and brittle spelling. It does not equalize languages: byte-level encodings still differ in length by more than four times for some language pairs [R110]. Against that, serving kernels, caches, and tools are built around token sequences. As of this edition, the model reports cited in this chapter, from PaLM to Llama 3, Gemma 3, and OLMo 2, still use learned tokenizers, and the tokenizer-free results above come from research-scale models.

::: practice Tokenizer experiment
Take the same sentence in several languages, plus an email address, a decimal, emoji, and code. Inspect token IDs and boundaries with the tokenizer of a model you use. Record token count and surprising splits. This one experiment cures many false intuitions about “what the model sees.”
:::

# Logits, Softmax, and the Next-Token Objective

The final model layer produces one score per vocabulary token. These unnormalized scores are **logits**.

## Softmax

For logits **z**, softmax produces probabilities:

`softmax(z_i) = exp(z_i) / Σⱼ exp(z_j)`

Exponentiation makes larger differences matter; division normalizes the values to sum to one. Adding the same constant to every logit does not change probabilities. Implementations subtract the maximum logit before exponentiating to prevent overflow.

Example logits `[2, 1, 0]` become probabilities approximately `[0.665, 0.245, 0.090]`. The ranking is unchanged, but differences become probabilistic.

## Output projection

If hidden state **h** has `d_model` features and output matrix **W_vocab** has shape `[d_model, vocabulary]`, then:

`logits = h W_vocab`

Some models tie this matrix to the input embedding matrix, reducing parameters and sharing geometry. Others use separate or factored heads.

## Teacher forcing and shifted targets

Given tokens:

```text
<BOS> the sky is blue <EOS>
```

training inputs and targets are shifted:

```text
input:  <BOS> the  sky  is    blue
target: the   sky  is   blue  <EOS>
```

The model predicts every next token in parallel during training because the full target sequence is known and causal masking prevents looking ahead. During generation, the unknown next token must be selected and fed back, making decoding sequential.

This mismatch is sometimes called exposure bias: training conditions on true prefixes, while inference eventually conditions on its own possibly mistaken outputs.

## Cross-entropy gradient intuition

For softmax plus one-hot cross-entropy, the gradient with respect to a logit is beautifully simple:

`∂L/∂z_i = predicted_probability_i - target_indicator_i`

The target token’s logit is pushed upward; other logits are pushed downward in proportion to their assigned probabilities. Earlier layers receive this signal through the output projection and chain rule.

## What next-token loss rewards

It rewards assigning probability to continuations found in the data. To do that well across diverse text, a model can benefit from learning:

- spelling and syntax;
- discourse and style;
- factual associations;
- code patterns and algorithms;
- latent variables such as speaker, topic, or genre;
- approximations to reasoning procedures that predict worked solutions.

But the objective also rewards reproducing misconceptions, biases, private strings, and persuasive rhetoric when present. It does not distinguish a factual sentence from a plausible false one unless context and learned statistics support the difference.

## Beyond one next token at a time

Next-token prediction is the base objective, but two modifications now appear in published work.

**Multi-token prediction (definition).** Instead of predicting only token `t+1` from position `t`, add output heads that also predict `t+2`, `t+3`, and so on. A 2024 study attached `n` independent output heads, each a transformer layer, to one shared trunk with a shared output matrix. The loss sums the cross-entropy over the future offsets. At inference they used only the next-token head, or used the extra heads to speed up decoding [R118]. The authors reported better sample efficiency that grew with model size, especially on code, and decoding up to three times faster with four-token prediction [R118]. DeepSeek-V3 adopted a multi-token prediction objective with an implementation that keeps the complete causal chain at each prediction depth [R66]. The added heads cost extra parameters and compute unless layers are removed elsewhere to compensate, as the 2024 study did to keep parameter counts equal while reporting no training-time overhead. The reported gains depended on scale: at small sizes its models did worse than the baseline. Part III, “Multi-token prediction,” covers the decoding use.

**Masked diffusion language models (definition and early evidence).** Autoregressive models factor a sequence’s probability left to right, as the chain rule in Part I describes. A masked-diffusion model is trained differently. Sample a mask ratio `t` uniformly between 0 and 1, independently replace each token with a special mask symbol with probability `t`, and train a transformer *without a causal mask* to recover the original tokens at the masked positions. The loss is the cross-entropy over masked positions, divided by `t`, and it upper-bounds the negative log-likelihood, so it is a principled generative objective [R119]. The division by `t` means a lightly masked sample, say `t = 0.25`, counts each masked token four times as heavily as a token in a fully masked one. BERT used a fixed masking ratio. Because `t` ranges up to 1, this objective includes fully masked inputs, so the model learns to produce text from nothing: to generate, start from a fully masked sequence and fill positions in over several steps, in any order. LLaDA trained models of 1 and 8 billion parameters from scratch this way. Its authors report results comparable to autoregressive baselines they trained themselves, and competitive in-context learning against a similarly sized open autoregressive model [R119]. This is early evidence from one group at one scale. Part III, “Diffusion language models,” covers the architecture, key–value cache, and decoding consequences.

## Base models and assistant behavior

A base model asked “What is the capital of…” may complete a quiz, continue a web page, imitate a debate, or answer directly depending on its learned continuation distribution. An assistant is created by additional training and a runtime that make instructions and roles more predictable.

This distinction explains many debates. “The model only predicts the next token” describes the local training objective. It does not fully characterize the algorithms and representations that minimizing that objective can produce, nor the complete system around inference.

::: idea Levels of explanation
A chess engine “only flips bits,” “only evaluates arithmetic,” and “searches a game tree.” All are true at different levels. Likewise, an LLM predicts tokens, executes learned tensor computations, and may implement useful higher-level procedures. Good explanations connect levels instead of using one to dismiss the others.
:::
