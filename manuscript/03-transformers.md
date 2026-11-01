# Part III — The Transformer, Piece by Piece

The transformer is a reusable architecture for routing and transforming information. Its central operation, attention, lets each position construct a context-dependent mixture of other positions. The rest of the block provides nonlinear computation, stable optimization, and depth.

One thread runs through this part. The 2017 block is still recognizable in 2026 models, but much of the architectural change since 2023 targets two costs: the memory and bandwidth needed to remember the past (the key–value cache), and the compute and memory of the feed-forward layers. When you meet a new architectural term, ask which of those two costs it reduces, what it gives up, and how it can fail.

# Attention from First Principles

Imagine the sentence:

> The animal did not cross the street because **it** was tired.

Interpreting “it” benefits from information about “animal,” not simply the nearest noun. We want a learned mechanism that lets the representation at one position gather relevant content from other positions.

## Query, key, and value

A database metaphor helps:

- a **query** describes what the current position seeks;
- a **key** describes what each source position offers or how it should be matched;
- a **value** contains information to retrieve if that source is relevant.

For input matrix **X** with one row per token, learned projections create:

`Q = XW_Q`

`K = XW_K`

`V = XW_V`

Each row of **Q** is a query, each row of **K** a key, and each row of **V** a value. Calling them query/key/value does not assign fixed human meanings; training discovers useful projections.

## Scores

Compare every query with every key using a dot product:

`S = QKᵀ`

If there are `T` positions, **S** has shape `[T, T]`. Entry `Sᵢⱼ` says how strongly destination position `i` matches source position `j` under this head.

As key dimension `d_k` grows, random dot products tend to grow in magnitude. If the components of a query and a key are independent with mean 0 and variance 1, their dot product has mean 0 and variance `d_k`, so its typical size is about `sqrt(d_k)` [R07]. Large logits make softmax extremely peaked and gradients less useful. Scaled dot-product attention divides scores by `sqrt(d_k)` [R07], which restores unit variance under the same assumption:

`S = QKᵀ / sqrt(d_k)`

## Masks

A mask marks disallowed connections. In a causal decoder, position `i` may attend only to positions `j ≤ i`. Future scores receive negative infinity before softmax, giving them zero probability.

Padding masks prevent attention to padding. Packed-sequence masks prevent unrelated documents in one tensor from seeing one another. Local, block-sparse, or sliding-window masks restrict connectivity for efficiency or inductive bias.

Masks are part of semantics, not just optimization. One incorrect mask can leak target answers during training or information between users.

## Normalize and mix

Apply softmax across each query row:

`A = softmax(S + mask)`

Every row of **A** sums to one over allowed sources. Then combine values:

`O = AV`

This is attention. In one expression:

`Attention(Q,K,V) = softmax(QKᵀ/sqrt(d_k) + mask)V`

The output at each position is a weighted average of projected source values.

## A tiny numerical example

Suppose one query has scores `[2.0, 1.0, -∞]` after scaling and masking. Softmax gives approximately `[0.731, 0.269, 0]`. If the first two value vectors are `[1, 0]` and `[0, 2]`, the output is:

`0.731[1,0] + 0.269[0,2] = [0.731, 0.538]`

The third value contributes nothing because the mask excluded it.

## Attention is content-dependent routing

Unlike a fixed convolution kernel, weights depend on the current hidden states. The same token can attend differently in different contexts. One head may often connect delimiters, another track nearby syntax, another copy a name, and many defy simple interpretation.

Attention weights are not automatically explanations. A high weight indicates contribution through one weighted-sum path, but outputs also depend on values, output projections, residual streams, later layers, and nonlinear interactions. Causal explanation requires interventions, not heatmaps alone.

# Multi-Head Attention

One attention operation has one similarity space and one value space. Multi-head attention performs several smaller attention operations in parallel.

## Split into heads

Let model width be `d_model`, number of heads `h`, and per-head dimensions often `d_k = d_v = d_model/h`. The projections produce head-indexed tensors:

- **Q**: `[batch, heads, sequence, d_k]`
- **K**: `[batch, heads, sequence, d_k]`
- **V**: `[batch, heads, sequence, d_v]`

Each head computes attention independently. Outputs concatenate and pass through learned projection **W_O**:

`MHA(X) = Concat(head₁, …, head_h)W_O`

Different heads can route different relationships simultaneously. Heads are not guaranteed to become clean linguistic modules; redundancy and distributed computation are common.

The rule `d_k = d_v = d_model/h` is a convention, not a requirement. One open-weight model reports a residual width of 2,880 with 64 query heads of dimension 64, which is 4,096 query features in total [R120]; another uses 16 query heads of dimension 256 over a residual width of 2,048 [R121]. What must match is only the shape attention needs: queries and keys share a dimension, and **W_O** maps the concatenated head outputs back to `d_model`.

## Grouped and multi-query attention

During generation, storing keys and values for every layer, position, and head consumes memory; the section on the key–value cache below counts it. Two designs cut that cost by letting several query heads share keys and values:

- **Multi-query attention (MQA)** keeps every query head but uses a single key head and a single value head, shared by all of them [R122].
- **Grouped-query attention (GQA)** divides the `h` query heads into `g` groups and gives each group its own key head and value head. `g = 1` is MQA; `g = h` is ordinary multi-head attention [R22].

With GQA, **Q** keeps shape `[batch, h, sequence, d_k]`, but **K** and **V** have shape `[batch, g, sequence, d_k]`, and each group of `h/g` query heads reads the same keys and values. The cache shrinks by a factor of `h/g`: 64 query heads sharing 8 key/value heads store one eighth as much. The key and value projections shrink by the same factor, so the attention block has fewer parameters than the `4d²` of full multi-head attention; for 64 query heads of dimension `d/64` sharing 8 key/value heads it has `2.25d²` (computed).

The two designs do not carry the same quality risk. In the GQA paper, a single key/value head could lower quality and make training less stable: T5-Large models trained from scratch with multi-query attention showed frequent loss spikes and diverged when fine-tuned on long-input tasks, while uptrained models with grouped attention appeared stable [R22]. A moderate number of groups gave quality close to full multi-head attention with speed comparable to MQA [R22]. An existing multi-head checkpoint can be converted by mean-pooling its key and value heads within each group and training briefly, using about 5% of the original pretraining compute [R22]. These are empirical findings on the models studied, not guarantees for every scale.

GQA is now a common default in open-weight dense attention. Reports describe 8 key/value heads in Llama 3 [R112], 32 query heads over 8 key/value heads in Mistral 7B [R123], 64 over 8 in gpt-oss [R120], and GQA in Qwen3, Gemma 3, and Command A [R124] [R101] [R125]. Some hybrid models go further: 16 query heads over 2 key/value heads in Qwen3-Next’s attention layers [R121] and 32 over 2 in Nemotron 3 Super [R126].

Query heads remain numerous, preserving multiple query subspaces; grouped keys and values make serving cheaper.

## Multi-head latent attention

GQA saves memory by using fewer key/value heads. **Multi-head latent attention (MLA)**, introduced with DeepSeek-V2, takes a different route [R127]. Every query head still gets its own keys and values at attention time, but the model does not store them. It stores one small vector per token from which they can be rebuilt.

The idea is *low-rank compression*. Mapping a hidden state to the keys and values of all heads is a large matrix multiplication, and it can be factored through a narrow bottleneck. Let `x_t` be the hidden state of token `t`:

`c_t = W_DKV x_t`  (down-projection to a latent of size `d_c`)

`k_t = W_UK c_t`  (up-projection to the keys of all heads)

`v_t = W_UV c_t`  (up-projection to the values of all heads)

Here `d_c` is much smaller than the combined key width `h · d_k`. Only `c_t` is cached, so each token and layer stores `d_c` numbers instead of `2 · h · d_k` [R127].

It looks as though every cached token would still need both up-projections at every step. It does not. A score between a query and a cached key is `(W_Q x)ᵀ(W_UK c)`, which equals `(W_UKᵀ W_Q x)ᵀ c`. The key up-projection can therefore be merged into the query side, so the query is compared directly with the cached latents; the value up-projection can likewise be merged into **W_O** [R127]. During decoding this makes MLA behave like multi-query attention over the latent: every query head reads the same cached vector per token [R128].

Rotary position embeddings complicate this. A rotation that depends on the key’s position would sit between the two matrices and prevent the merge. MLA therefore splits each query and key into a content part, which uses the latent, and a small *decoupled* rotary part, one extra shared key per token that carries position. The cache holds the latent plus that rotary key, `d_c + d_R` numbers per token and layer [R127].

A real configuration shows the size of the effect. DeepSeek-V3 uses 128 heads of dimension 128, `d_c = 512`, and a rotary dimension of 64, so its cache needs `512 + 64 = 576` numbers per token and layer [R66]. Plain multi-head attention with the same heads would need `2 × 128 × 128 = 32,768`, about 57 times more (computed). DeepSeek-V2 compared its own setting to GQA: the MLA cache equals that of GQA with 2.25 groups, and the paper reports a 93.3% smaller cache than the same lab’s previous dense model [R127]. The V2 paper also reports MLA performing better than multi-head attention in its ablations; that is the authors’ result on their own recipe.

MLA trades memory for arithmetic and complexity:

- Decoding performs a `d_c + d_R`-dimensional dot product per head and cached token, more than the 128 dimensions of a typical GQA head. One lab reported that its MLA variant with a 576-dimensional latent did not match 8-group GQA under its optimizer until the optimizer was adjusted, and then made heads fewer and wider to cut decoding compute [R129].
- Another lab halved the heads from 128 to 64: at a 128,000-token context, doubling heads raised inference compute by 83% for a 0.5%–1.2% lower validation loss [R93].
- Normalizing queries and keys to tame attention logits (a stability tool covered in Part II, “Keeping large training runs stable”; Qwen3 and Gemma 3 use it [R124] [R101]) cannot be applied directly to MLA, because the full keys are never materialized at inference. One lab instead clipped the query and key projection weights when logits grew [R93].

MLA appears in the DeepSeek-V2, V3, and V3.2 models, in Kimi K2 and K3, and in GLM-5 [R127] [R66] [R128] [R93] [R130] [R129].

## Cross-attention

Self-attention derives Q, K, and V from the same sequence. Cross-attention derives queries from one stream and keys/values from another. An encoder–decoder model lets decoder states query encoder outputs. A multimodal model may let text queries attend to encoded image or audio features.

# The Decoder-Only Transformer Block

Most generative LLMs use a stack of causal decoder blocks. A common pre-norm block is:

```text
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

Repeat this `L` times, apply a final norm, then project to vocabulary logits.

## Full data flow

For token IDs `[batch, sequence]`:

1. look up token embeddings → `[batch, sequence, d_model]`;
2. if the model uses learned absolute or sinusoidal position embeddings, add them here; with rotary embeddings, the common choice today, nothing is added at this point, because position enters in step 3 by rotating queries and keys inside attention;
3. normalize and compute masked multi-head self-attention, applying the rotary rotation to queries and keys if used;
4. add the attention result to the residual stream;
5. normalize and compute a position-wise gated MLP (or, in a mixture-of-experts model, a routed set of expert MLPs);
6. add the MLP result;
7. repeat through all blocks;
8. normalize final hidden states;
9. project each position to vocabulary logits;
10. compute next-token loss during training or decode one next token during inference.

Attention communicates across positions. The MLP transforms features within each position. Depth alternates communication and computation.

## Parameter accounting

For a dense block with width `d` and MLP expansion around `4d`, rough parameter counts are:

- attention Q, K, V, O projections: about `4d²` for full multi-head attention, less when keys and values are shared (see grouped-query attention above);
- two ordinary MLP projections: about `8d²`;
- gated MLP may use about `12d²` before choosing a smaller expansion;
- normalization parameters are only `O(d)`.

Multiply by layer count, then add embeddings/output head. Architecture variants alter the constants. Parameter count describes storage but not active compute, training tokens, data quality, or capability.

## Residual stream as shared workspace

A helpful abstraction treats the residual stream as a high-dimensional workspace. Attention heads read from it through Q/K/V projections and write routed updates through **W_O**. MLPs read features and write transformed features. Later layers can combine previous updates.

The workspace analogy explains why residual connections help compositional depth. Do not take it literally as a list of symbolic variables; features may overlap and use superposition.

## Why depth helps

A single attention layer can gather information, but deeper layers can create multi-step computations:

- early layers identify local patterns;
- one layer routes a relevant name;
- an MLP converts it into a useful feature;
- later attention routes that feature to an answer position;
- later MLPs transform it into logits.

Actual circuits can be distributed and task-dependent. Depth increases sequential computational steps per token even though operations within a layer are parallel across training positions.

# Position, Context, and Long Sequences

Unmasked self-attention without position is permutation-equivariant: shuffling tokens shuffles outputs but does not tell the model order. A causal mask leaks order only implicitly, through how many tokens each position can see. Position mechanisms break that symmetry explicitly.

## Rotary position embeddings

RoPE [R20] groups query and key features into pairs and rotates each pair by a position-dependent angle. For one pair, with `θ` equal to the token’s position times that pair’s frequency:

```text
[x1']   [ cos θ  -sin θ ][x1]
[x2'] = [ sin θ   cos θ ][x2]
```

Different feature pairs use different frequencies. In the original formulation the frequency of pair `j`, counting from 0 across a head of dimension `d`, is `10000^(-2j/d)` [R20], so frequencies fall geometrically from 1 toward roughly 1/10,000. A pair with frequency `f` turns a full circle every `2π/f` tokens; call that its *wavelength*. For a head of dimension 128 (64 pairs) and base 10,000, wavelengths run from about 6 tokens to about 54,000 tokens (computed).

The dot product between rotated queries and keys depends on relative position: rotating the query by `mθ` and the key by `nθ` leaves a net rotation by `(n−m)θ` inside the product [R20]. RoPE is efficient and widely used, but extrapolating beyond trained lengths requires care.

The frequencies do different jobs. In one analysis of a trained model, attention heads used the highest frequencies for sharp position-based patterns and the lowest for matching that barely depends on distance [R131]. A pair whose wavelength exceeds the longest training sequence never completes a turn during training. The YaRN authors argue that such a pair can act as an absolute position marker, and that it meets unfamiliar angles when the context grows [R132].

## Extending context after pretraining

Long sequences are expensive to train on and long documents are scarce, so models are usually pretrained mostly on short sequences and then extended in a late stage. Llama 3 405B grew from 8,000 to 128,000 tokens in six stages near the end of pretraining [R112]; DeepSeek-V3 went from 4,000 to 32,000 to 128,000 tokens in two stages of 1,000 steps each [R66]; the 4B, 12B, and 27B Gemma 3 models pretrained on 32,000-token sequences and then rescaled RoPE by a factor of 8 to reach 128,000, and the report notes that quality degrades rapidly beyond that length [R101].

Four ideas appear in these recipes:

- **Position interpolation** divides all position indices by the extension factor, so that, for example, a 32,768-token input uses angles the model saw within 4,096 tokens (factor 8). Extrapolating to unseen positions can produce very large attention scores; interpolating avoids that, and the original paper reached a 32,768-token window with minimal fine-tuning, within 1,000 steps [R133]. The price is resolution: neighboring tokens now differ by a smaller angle, which blurs fine local order in the fast-rotating pairs [R132].
- **Raising the base frequency** stretches all wavelengths. Llama 3 uses a base of 500,000 [R112]; Qwen3 raised it from 10,000 to 1,000,000 in its long-context stage [R124] [R134]; Gemma 3 uses 1,000,000 in its global layers and 10,000 in its local ones [R101].
- **YaRN** treats each pair according to its wavelength [R132]. A pair that turns many times within the training length (a short wavelength) keeps its frequency, because it encodes local order. A pair that turns less than once is interpolated by the full factor. Pairs in between are blended by a linear ramp. The cutoffs are set by how many turns a pair makes over the original length: no change above 32 turns, full interpolation below 1 turn, the values the authors found for the LLaMA family. The recipe also scales the attention logits slightly: the authors fitted, on LLaMA models, a multiplier of `0.1 ln(s) + 1` on both queries and keys for extension factor `s`, which multiplies the logits by about 1.46 for `s = 8` (computed). For a 128-dimensional head with base 10,000 trained at 4,096 tokens, these cutoffs leave 21 of the 64 pairs unchanged, interpolate 18, and blend 25 (computed). DeepSeek-V3 used `s = 40` with those cutoffs, applied only to its decoupled rotary key [R66]; gpt-oss extends its dense layers to 131,072 tokens with YaRN [R120].
- **Non-uniform search.** LongRoPE searches for a separate rescaling per dimension and extends in stages, reporting windows of 2,048k (2,097,152) tokens for extended LLaMA2 and Mistral models [R135].

These are empirical recipes with tuned constants, not theorems. Position scaling changes geometry; it cannot create missing long-range training experience, so the strongest results above also involve at least brief training on longer sequences.

## Not every layer needs a rotation

Because a causal mask already leaks order, a decoder can work with no explicit position signal at all (called NoPE). In one controlled study of decoder-only transformers, NoPE generalized to longer inputs than several explicit schemes, and the authors argued it can learn something resembling relative position [R136]. Recent large models mix the options layer by layer:

- Command A interleaves three sliding-window layers that use RoPE with one full-attention layer that uses no positional embedding [R125].
- Kimi Linear and Kimi K3 give no positional encoding to their global attention layers and leave position to the recurrent layers between them [R137] [R130]. Nemotron 3 Super omits positional embeddings in its attention layers too [R126].
- **p-RoPE** keeps only a fraction `p` of the rotary frequency pairs, dropping the lowest frequencies; `p = 0` is NoPE [R131]. Gemma 4 uses `p = 0.25` in its global layers and ordinary RoPE in its local layers [R138]. DeepSeek-V4 rotates only the last 64 dimensions of each query and key [R139].

Which combination is best for very long contexts is still an open empirical question; the reports above each justify their choice with experiments at their own scale.

## Effective versus advertised context

A model may accept `N` tokens without reliably using every token. Effective context depends on:

- length distribution during training and post-training;
- position encoding and scaling;
- task type and location of relevant evidence;
- distractors and repeated patterns;
- retrieval or memory formatting;
- inference precision and serving implementation.

“Needle in a haystack” tests check whether a model can retrieve a planted fact, but real long-context tasks require integrating, comparing, and reasoning over many pieces. Evaluate the actual workload.

Two measured examples show why. In a 2023 study, accuracy was often highest when the relevant passage sat at the beginning or end of the context and dropped significantly when it sat in the middle [R140]. In 2024, a benchmark that extends needle tests with multiple needles, multi-hop tracing, and aggregation found that although models scored nearly perfectly on the simple needle test, almost all showed large drops as length grew, and only about half of the 17 models tested, all claiming 32,000 tokens or more, kept satisfactory performance at 32,000 [R141]. Those results describe 2023–2024 models; they motivate testing, not a verdict on today’s.

A window can also be advertised beyond the length a model was trained on. One 2025 open-weight release announced a 10-million-token window for one variant while reporting that it was pretrained and post-trained at 256,000 tokens, relying on length generalization from interleaved layers without positional embeddings and from temperature scaling of attention at inference [R142].

## Context lengths in late 2026

As of 2 November 2026, the API documentation of three large providers lists windows of about one million tokens for current models: 1.05 million tokens in OpenAI’s model listing, 1 million in Anthropic’s documentation (some older models still list 200,000), and 1,048,576 in Google’s Gemini API documentation [R143] [R144] [R145]. Output limits are much smaller: the same pages list maximum single responses of 128,000 tokens at OpenAI and Anthropic, and Google lists 65,536 tokens for its Gemini 3 and 2.5 models. Google’s newest documented family counts input, reasoning, and output tokens against one shared window, with a 65,536-token default response cap [R145].

Open-weight reports show the same range. DeepSeek-V4, Nemotron 3 Super, and Kimi K3 document one million tokens, Qwen3.5 lists 262,144 tokens natively and extensibility to about 1,010,000, Gemma 4 lists 128,000 or 256,000 depending on size, and gpt-oss uses 131,072 [R139] [R126] [R130] [R146] [R147] [R120]. A window is an upper bound on what the interface accepts. Whether the model uses it well is a separate measurement (see the previous section). The cache arithmetic below shows why a million-token window needs more than rescaled positions.

## Quadratic cost

Standard attention constructs `T × T` scores. Training compute for attention therefore grows quadratically with sequence length, while MLP cost grows linearly in `T` for fixed width. A naive implementation also stores the `T × T` score matrix, so its memory is quadratic too; exact IO-aware kernels such as FlashAttention avoid storing it, so the extra memory becomes linear in `T` while the arithmetic stays quadratic [R19]. At large width and moderate context, projections and MLPs may dominate; at very long context, attention becomes critical.

Efficiency approaches include:

- memory-efficient exact kernels such as FlashAttention [R19];
- local or sliding-window attention, and interleaving local with global layers;
- sparse global tokens, block patterns, or learned sparse selection;
- compressing, sharing, or reducing the key–value cache;
- chunking and recurrence;
- compression or learned memory;
- distributing sequence blocks across devices;
- alternative sequence models with linear-time state updates.

“Linear attention” often changes the computation or approximation; check quality, causal support, training behavior, and hardware efficiency rather than complexity notation alone. Later sections take up the window, sparse, and recurrent options in turn.

## FlashAttention

FlashAttention computes exact attention while tiling operations so large intermediate score matrices need not be repeatedly written to slow high-bandwidth memory. It exploits the memory hierarchy, reduces data movement, and recomputes small pieces in the backward pass [R19].

This is an important systems lesson: the mathematical operation can remain unchanged while an IO-aware algorithm makes it much faster. Later versions kept the same exact attention: FlashAttention-2 improved parallelism and work partitioning [R148], and FlashAttention-3 exploited asynchrony and low-precision arithmetic on newer GPUs [R149]. The serving chapter returns to these kernels.

# Training Parallelism Inside the Architecture

The transformer removed recurrent dependence between training positions: with causal masks, all positions’ next-token predictions can be computed in parallel because targets are known. This does not mean the model ignores order; the mask and positions preserve it.

Generation remains sequential across new tokens:

```text
prefix → token 1
prefix + token 1 → token 2
prefix + token 1 + token 2 → token 3
```

The model can process multiple requests or candidate sequences in parallel, but each individual autoregressive chain has a dependency.

This distinction—parallel training, sequential decoding—shaped the economics of LLMs.

# The Key–Value Cache

Without caching, generating each token would recompute keys and values for the entire prefix at every layer. During causal generation, representations of previous positions at a given layer do not change. We cache their keys and values; the stored collection is the **key–value cache**, or KV cache.

## Cache operation

At each new token and layer:

1. compute its query, key, and value;
2. append the new key and value to that layer’s cache;
3. compare the new query with all cached keys;
4. mix cached values;
5. continue through the block.

A rough cache size in bytes for ordinary attention is:

`2 × layers × sequence × kv_heads × head_dim × bytes_per_value`

The factor 2 stores keys and values. Multiply by concurrent sequences. GQA, MQA, quantized caches, paging, eviction, and prefix sharing reduce cost.

Each term in that product is a lever, and most 2024–2026 attention designs pull exactly one:

- fewer `kv_heads`: multi-query and grouped-query attention;
- a smaller entry per token: multi-head latent attention replaces `2 × kv_heads × head_dim` by `d_c + d_R`;
- fewer cached `sequence` positions: sliding-window layers keep only the last `window` tokens, and compressed or sparse attention stores or reads fewer entries;
- fewer `layers` that keep a growing cache: recurrent layers hold a fixed-size state, and some designs let layers share one cache;
- fewer `bytes_per_value`: quantized caches, and mixed formats such as a low-precision value for most dimensions and higher precision for the rotary ones [R139];
- paging and prefix reuse manage the cache rather than shrinking it [R46].

## A worked comparison

The code below sizes the cache of one sequence under five designs. The model is a plain illustration, not any released model: 60 layers, 64 query heads of dimension 128, 131,072 tokens, 2 bytes per value, ignoring weights, activations, and paging overhead.

```python
def kv_cache_bytes(layers, tokens, kv_heads, head_dim, bytes_per_value=2, window=None):
    """Keys and values for one sequence; a sliding window caps how many tokens are kept."""
    kept = tokens if window is None else min(tokens, window)
    return 2 * layers * kept * kv_heads * head_dim * bytes_per_value


def mla_cache_bytes(layers, tokens, latent_dim, rope_dim, bytes_per_value=2):
    """One shared latent vector plus one shared RoPE key per token and layer."""
    return layers * tokens * (latent_dim + rope_dim) * bytes_per_value


T = 131_072  # tokens in the context
GB = 1e9

mha = kv_cache_bytes(60, T, kv_heads=64, head_dim=128)
gqa = kv_cache_bytes(60, T, kv_heads=8, head_dim=128)
mqa = kv_cache_bytes(60, T, kv_heads=1, head_dim=128)
mla = mla_cache_bytes(60, T, latent_dim=512, rope_dim=64)
# five sliding-window layers (1,024 tokens) for every global layer, all with 8 KV heads
local_global = (kv_cache_bytes(50, T, 8, 128, window=1024)
                + kv_cache_bytes(10, T, 8, 128))

for name, value in [("MHA, 64 KV heads", mha), ("GQA, 8 KV heads", gqa),
                    ("MQA, 1 KV head", mqa), ("MLA, 512 + 64", mla),
                    ("GQA 8, five local layers per global layer", local_global)]:
    print(f"{name:45s} {value / GB:7.2f} GB")
```

Running it prints:

```text
MHA, 64 KV heads                               257.70 GB
GQA, 8 KV heads                                 32.21 GB
MQA, 1 KV head                                   4.03 GB
MLA, 512 + 64                                    9.06 GB
GQA 8, five local layers per global layer        5.58 GB
```

Two consequences follow. First, a long-context cache can exceed the weights of a modest model: a 7-billion-parameter model stored in 16-bit precision has about 14 GB of weights, less than the 32 GB cache above, and the cache multiplies by the number of concurrent sequences. Second, decoding must read the cache of the full-attention layers once per generated token. If a device streams 3 terabytes per second (an assumed round number), reading a 32.2 GB cache takes at least 10.7 ms per token, about 93 tokens per second for one sequence before counting the weights (computed). That is why decode throughput depends so heavily on cache size.

At one million tokens the same GQA configuration needs 245.8 GB per sequence (computed). That arithmetic is the reason the next chapter exists.

## Layers that share a cache

Layers need not each keep their own. Cross-layer attention lets neighboring layers reuse one set of keys and values [R150]. YOCO computes the keys and values once in the lower half of the network and lets the upper half read them, so prefill can exit early [R151]. Gemma 4’s two smallest models share keys and values across layers, in 20 of 35 layers and 18 of 42 layers respectively [R138] [R147]. DeepSeek-V4.1-Flash builds its decoder’s global keys and values from the hidden state at the midpoint of the stack and reuses compressed entries and index choices across layers, reporting about 890 bytes of global cache per token, roughly a quarter of its predecessor’s, with the main cache stored in 4-bit floating point [R152]. Reported numbers like these are measured by the authors against a baseline of their choosing; compare the mechanism, not the headline ratio.

## Prefill and decode

Serving has two phases:

- **Prefill:** process the complete prompt in parallel and build caches. Compute-heavy for long prompts.
- **Decode:** generate tokens one at a time. Often memory-bandwidth- and latency-sensitive.

Time to first token depends heavily on prefill and queueing. Time per output token reflects decode. Throughput optimization may increase latency for one user by batching more users, so service-level objectives matter.

::: idea The cache is the design constraint
Many 2024–2026 attention changes look unrelated until you read them as answers to one question: how can a model keep a useful memory of a million tokens without storing a million full-width key–value entries in every layer? Grouping, latent compression, windows, sparsity, sharing, quantization, and recurrent state each answer part of it, and each pays in quality, compute, or engineering complexity.
:::

# Cheaper Attention for Long Sequences

Three families of design make attention cheaper for long contexts:

1. restrict which positions a query may see by a fixed rule (windows and local–global patterns);
2. restrict it by a learned, content-dependent choice (sparse attention);
3. replace the growing cache with a fixed-size state (linear attention and state-space layers, covered under architectural variants below).

Attention sinks, a side effect of softmax, matter for all three.

## Sliding-window and local–global layers

In **sliding-window attention**, each position attends only to the previous `W` positions. Longformer used windows, with a few global positions, to make attention cost linear in length [R153]. Mistral 7B used a window of 4,096 tokens in every layer [R123]. Information can still travel further than `W`: stacked layers pass it forward, so after `k` layers a token can in principle be influenced by tokens up to `k × W` positions back, about 131K for 32 layers with `W = 4,096` [R123]. A later report, citing earlier work, notes that the effective reach of a sliding window is much smaller than this theoretical bound [R152]. Only the last `W` keys and values are needed, so the cache lives in a rolling buffer of fixed size; Mistral reports an 8× smaller cache on a 32K-token sequence [R123].

A window alone cannot retrieve a fact that lies far outside it. Modern designs therefore **interleave** local and global layers. Gemma 3 uses five local layers with a 1,024-token window for every global layer. In its ablations, perplexity changed little across local-to-global ratios up to 7:1 and with much shorter windows, and for a 2B-parameter test model with a 32K-token context the memory overhead of the KV cache was about 60% with global attention only, but under 15% with three local layers per global layer and 1,024-token windows [R101]. gpt-oss alternates banded layers of width 128 with dense layers [R120]. Command A uses three window layers to one full layer [R125]. Gemma 4 keeps the 5:1 pattern (4:1 in its smallest model) and lets global layers in the larger models reuse keys as values [R138]; its model card lists 1,024-token windows for the larger models [R147]. DeepSeek-V4 adds a small sliding-window branch alongside compressed global attention, so each query sees recent tokens exactly [R139]. With 60 layers, 5:1 interleaving, 1,024-token windows, and 8 key/value heads, the worked example above falls from 32.2 GB to 5.6 GB at 131,072 tokens (computed).

Interleaving is not free. The global layers must still do the long-range retrieval, and one lab reported that adding sliding-window layers to its model degraded quality noticeably as context grew, so it kept full attention [R154]. The ratio and window that work depend on the task mix and on the model’s other layers.

## Attention sinks

Softmax forces each row of weights to sum to one. A head that finds nothing useful to read must still put its weight somewhere. StreamingLLM observed that models put a large share of attention on the first few tokens regardless of what those tokens say, and called them **attention sinks**. When a sliding window evicted those initial tokens from the cache, quality collapsed; keeping the keys and values of just the first four tokens alongside a window of recent ones restored stable perplexity over millions of tokens [R155]. The authors’ explanation is that the first tokens are visible to every later position, so they are easiest to train as a dump for surplus attention.

StreamingLLM does not extend what a model can use: it keeps generation stable on a stream, but it does not give the model access to evicted text [R155]. The same paper found that adding a dedicated placeholder token as a sink during pretraining can improve streaming further. Recent architectures build the sink in:

- gpt-oss gives each head a learned bias in the softmax denominator, so a head can attend to nothing [R120]. In symbols, weight `a_j = exp(z_j) / (exp(b) + Σ_k exp(z_k))`, where `b` is learned per head; the weights can then sum to less than 1.
- DeepSeek-V4 adds learned sink logits in the same way [R139].
- A sigmoid gate applied per head to the attention output lets a head scale its contribution toward zero. Experiments on 1.7-billion-parameter dense and 15-billion-parameter mixture-of-experts models found that this gate improved quality and stability and reduced attention sinks [R156]. Kimi K3 adds a similar input-dependent output gate to its latent-attention layers [R130], and Qwen3-Next calls its attention layers “gated attention” [R121].

## Learned sparse attention

Windows use a fixed rule. **Sparse attention** lets the model choose which earlier tokens each query reads, from the content. Done naively it fails: scoring every token to choose the best ones costs as much as attention. Practical designs make the scoring cheap, compress the keys first, or choose blocks rather than single tokens.

**Block selection.** MoBA divides the context into blocks, scores each block for a query by the dot product of the query with the block’s mean key, and attends to the top few blocks plus the query’s own block, borrowing the routing idea from mixture-of-experts layers [R157]. **Native Sparse Attention (NSA)** runs three branches for each query and mixes them with learned gates: attention over compressed blocks, attention over fully detailed tokens in the blocks the compressed scores rank highest, and a local sliding window. Its selection works on blocks (contiguous reads suit GPUs) and is shared by the query heads of a GQA group, so their cache reads coincide, and the selection is trained end to end rather than imposed afterward [R158].

**DeepSeek Sparse Attention (DSA)** scores tokens with a small *lightning indexer*: for a query at position `t` and an earlier token `s`, `I(t,s) = Σ_j w(t,j) · ReLU(q(t,j) · k(s))` over a few indexer heads, and ordinary attention then runs over the top-`k` entries (2,048 in DeepSeek-V3.2) [R128]. The main attention drops from `O(L²)` to `O(L·k)` work; the indexer is still quadratic but much cheaper because it has few heads and runs in low precision [R128]. Training is staged. A dense warm-up trains only the indexer, for 2.1 billion tokens, to imitate the dense attention distribution; then the whole model trains with sparse selection for 943.7 billion tokens starting from a dense checkpoint [R128]. With `k = 2,048` at 131,072 tokens, the last query reads 1.6% of earlier tokens, and averaged over a whole causal sequence about 3.1% of query–key pairs are read (computed). The authors report no substantial loss in the long-context evaluations they ran; GLM-5 adopted the same mechanism and found long-context results close to its MLA baseline, ahead on some of its long-context checks and behind on one [R128] [R129].

**DeepSeek-V4** combines compression with selection [R139]. In *compressed sparse attention*, the cache entries of every 4 tokens are merged into one by a learned weighted sum, and a DSA-style indexer picks the top 512 (Flash) or 1,024 (Pro) of those entries for each query. In *heavily compressed attention*, every 128 tokens are merged into one entry and attention is dense over the much shorter result. The two kinds of layer are interleaved, each with a small sliding-window branch for recent tokens. At one million tokens, compressing by 4 leaves 250,000 entries per layer, of which at most 1,024 are read, 0.4% (computed). The report states that at that length the larger model needs 27% of the per-token compute and 10% of the cache of DeepSeek-V3.2, and about 2% of the cache of a BF16 grouped-query baseline with 8 key/value heads of dimension 128 [R139]. Those are the authors’ estimates for their stack.

What sparse attention gives up is recall of anything the selector misses. If the compressed or indexed score ranks a needed token low, that token is invisible to the query for that layer. Training the selector end to end, mixing in a local window, and keeping some full or heavily compressed layers are the mitigations used so far; none is a guarantee.

## What the reported numbers mean

::: warning Efficiency numbers are conditional
Cache sizes, compute ratios, and speedups in model reports are measured on the authors’ hardware, kernels, precision, batch sizes, and baseline. “10% of the cache” is meaningful only relative to a named baseline at a named length. Different reports choose different baselines, so ratios from different papers cannot be multiplied or compared directly.
:::

# Architectural Variants

The canonical transformer is a family, not one fixed blueprint.

## Mixture of experts

A mixture-of-experts (MoE) feed-forward layer contains many expert MLPs and a router that selects a small number for each token. If a model has 64 experts but activates 2, total parameters can be large while active compute per token remains closer to two experts [R23, R24].

The router outputs expert scores; top-k experts process the token; weighted outputs combine. Training needs load-balancing so a few experts do not receive everything. Distributed systems must move tokens to devices holding selected experts, making network topology important.

MoE trade-offs:

- more parameter capacity per active FLOP;
- larger weight memory and checkpoint size;
- routing and communication overhead;
- potential expert imbalance and failure modes;
- complex serving at small batch sizes.

“Parameter count” must distinguish total and active parameters.

### Shapes and a numerical example

For `n` tokens in a batch, the router produces a score matrix of shape `[n, E]` for `E` experts. Each token’s top-`k` experts run on that token alone, and their outputs are combined with weights derived from the scores. DeepSeek-V3 has 671 billion parameters in total and 37 billion active per token, about 5.5% [R66]. At roughly 2 floating-point operations per active parameter, one forward pass per token costs about 74 gigaflops, ignoring the attention over the context, instead of about 1.3 teraflops if all parameters were active (computed estimate). The saving applies to arithmetic; memory does not shrink, since all experts must be stored: about 671 GB at one byte per parameter, about 1.3 TB at two.

### Fine-grained and shared experts

Early MoE layers used a few experts each as large as the dense MLP. **Fine-grained experts** split each into several smaller ones and activate proportionally more of them, keeping compute the same but making the number of possible combinations explode: choosing 2 of 16 experts allows 120 combinations, while choosing 8 of 64 experts of a quarter of the size allows over 4.4 billion (computed). DeepSeekMoE proposed this, and also **shared experts**: a few experts that process every token, so that common knowledge need not be duplicated across the routed experts, which can then specialize [R159]. Scaling-law work finds that setting the expert size equal to the dense MLP size is suboptimal at almost any compute budget [R160].

Not every lab uses shared experts. DeepSeek-V3 has 1 shared and 256 routed experts with 8 routed experts active per token, and its first three layers are dense [R66]. Kimi K2 has 1 shared and 384 routed experts with 8 active [R93]. Qwen3’s mixture-of-experts models use 128 experts with 8 active and no shared expert [R124]. Kimi K3 uses two shared experts with 896 routed experts, 16 active [R130]. The design is still being tuned.

### Load balancing without an auxiliary loss

If the router sends most tokens to a few experts, the others learn little and the busy devices become the bottleneck. The classic remedy adds an auxiliary loss term that penalizes uneven expert usage [R23, R24]. Its drawback is that this term produces gradients unrelated to prediction, and a large one hurts quality [R161]. A related finding is that computing the penalty inside small micro-batches forces every sequence to spread over all experts and discourages specialization; computing it over the global batch relaxes that [R162]. Qwen3 uses the global-batch version [R124].

**Auxiliary-loss-free balancing** replaces the loss with a per-expert bias that is added to the router scores only when choosing which experts to use. After each training step, the bias of an overloaded expert is lowered and that of an underloaded expert raised by a small step; the weights used to mix expert outputs still come from the unbiased scores [R161] [R66]. For example, take four experts with affinities `[0.9, 0.8, 0.6, 0.5]` and learned biases `[−0.2, 0, 0, +0.3]`. Selection uses the sums `[0.7, 0.8, 0.6, 0.8]`, so experts 1 and 3 are chosen instead of 0 and 1; their mixing weights come from the original affinities 0.8 and 0.5 normalized to `[0.615, 0.385]` (computed). The bias is adjusted by counting loads, not by gradient descent. DeepSeek-V3 used this method with a very small sequence-level balance loss kept as a safeguard, and dropped no tokens during training or inference [R66]. DeepSeek-V4 kept the scheme and reports that loss spikes were tied to outliers in the MoE layers, which the routing itself seemed to worsen; it added stabilizers, including computing routing decisions from slightly older parameters [R139].

Kimi K3’s authors report that fixed-step bias updates stop being well behaved at about a thousand experts, and replace them with a rule that sets each bias from a quantile of the router margins so that every expert receives its target load [R130].

### Very high sparsity

Sparsity is often quoted as total experts divided by active experts, or total parameters divided by active parameters. Reported configurations span a wide range:

| Report | Routed experts, active per token | Total / active parameters |
|---|---|---|
| DeepSeek-V3 | 256, 8 (plus 1 shared) | 671B / 37B |
| Kimi K2 | 384, 8 (plus 1 shared) | 1.04T / 32.6B |
| gpt-oss-120b | 128, 4 | 116.8B / 5.1B |
| Qwen3-Next-80B-A3B | 512, 10 (plus 1 shared) | 80B / 3B |
| Nemotron 3 Super | 512, 22 | 120.6B / 12.7B |
| DeepSeek-V4-Pro | 384, 6 (plus 1 shared) | 1.6T / 49B |
| Kimi K3 | 896, 16 (plus 2 shared) | 2.8T / 104B |

Sources: [R66] [R93] [R120] [R121] [R126] [R139] [R130]. In these reports the active fraction ranges from roughly 3% to 11% of the parameters (Nemotron 3 Super is the high end), and the ratio of routed experts to active ones ranges from about 23 to 64 (computed).

Why go sparser? Kimi’s scaling experiments held active compute fixed and varied the number of experts; more experts lowered loss consistently, and to reach the same validation loss, a sparsity of 48 needed 1.69 times fewer floating-point operations than a sparsity of 8 [R93]. Another study found that for fixed constraints on total parameters or compute, there is an optimal sparsity rather than a monotone gain [R163]. Both are empirical fits in limited ranges.

Costs rise with sparsity. Every expert must be stored. Under decoding, a batch touches many different experts: with 384 experts, 8 active per token, and tokens routed independently and uniformly, a batch of 64 tokens touches about 284 of the 384 experts, 74% (computed), so weight reads do not shrink in proportion to active parameters at moderate batch sizes. Tokens must also be sent to the devices that hold their experts and returned. **LatentMoE** projects the token down to a smaller latent width for routing and expert computation, which cuts expert weight loads and all-to-all traffic by the ratio of widths, and spends the savings on more experts and more active experts per token; Nemotron 3 Super (512 experts, 22 active, latent width 1,024 over a model width of 4,096) and Kimi K3 use it [R126] [R130].

## Linear attention, state-space layers, and hybrids

Standard attention keeps every key and value because each new query needs a fresh softmax over all of them. Remove the softmax and the sum can be rearranged. Without it, the output for a query `q` is the sum over earlier tokens of `(kⱼ·q) vⱼ`, which equals `(Σⱼ vⱼ kⱼᵀ) q`. The bracketed matrix `S = Σⱼ vⱼ kⱼᵀ` has a fixed shape, `[d_v, d_k]`, however many tokens have been added. Each new token adds one outer product, `S ← S + v kᵀ`, and reading is one matrix–vector product, `o = S q`. That is **linear attention**: it turns a causal attention layer into a recurrent network whose state is a matrix, with constant cost and memory per generated token [R164]. The published version also uses feature maps and a normalizer, omitted here.

The price is capacity. A state of that shape can store only about `d_k` independent key–value associations; beyond that they collide, and exact retrieval degrades [R165]. Better recurrences manage the state:

- **Decay (gating).** `S ← α S + v kᵀ`, with a learned, input-dependent `α` between 0 and 1, forgets old content uniformly. This is the update the Mamba-2 family uses [R165]. Mamba made state updates input-dependent, the “selective” state space [R25]; Mamba-2 simplified the structure so that it could use larger states and run 2–8 times faster, and showed a duality between such models and a form of attention [R166].
- **Delta rule.** Before storing a value for a key, subtract what the state already predicts for that key: `S ← S (I − β k kᵀ) + β v kᵀ`. This overwrites instead of piling on. It equals one gradient step on the squared error between the state’s prediction `S k` and the new value `v` [R165].
- **Gated delta rule.** `S ← S · α (I − β k kᵀ) + β v kᵀ` combines both: `α → 0` clears memory quickly, and `α → 1` leaves only targeted overwrites [R165]. Kimi Delta Attention gives each key channel its own decay instead of one scalar [R137].

A small example with a 2×2 state, `β = 1`, and `k = [1, 0]` makes the difference concrete (computed). Write value `[1, 2]`; reading with `k` returns `[1, 2]`. Now write `[5, 6]` under the same key. Additive linear attention returns `[6, 8]`, a mixture. The delta rule returns `[5, 6]`: the old association was replaced. Finally write value `[3, 4]` under an orthogonal key with decay `α = 0.5`: reading the first key now returns `[2.5, 3]`, half of what it was, so the gate has faded the older memory.

Fixed-size memory has a limit. One theoretical analysis proves that a two-layer transformer can copy strings of exponential length, while models with a fixed-size latent state are fundamentally limited by that size, and experiments on copying tasks agree [R167]. In an 8-billion-parameter comparison trained on the same data, pure state-space models matched or beat transformers on many tasks but lagged on tasks needing copying, in-context learning, or long-context reasoning [R168].

### Why hybrids

A hybrid keeps most layers recurrent, so their cache is a constant-size state, and inserts a few full-attention layers to provide exact recall of any earlier token. The same comparison found that a hybrid with about 43% Mamba-2 layers, 7% attention layers, and 50% MLP layers exceeded the pure transformer on all twelve standard tasks it evaluated, and its authors predicted much faster generation [R168]. In small-scale ablations, Jamba found that 1:3 and 1:7 ratios of attention to Mamba layers gave virtually the same quality, so its hybrid, which also uses mixture-of-experts layers, takes the cheaper 1:7 [R169]. A systematic study of linear layers in hybrids found that recall improved with more full-attention layers, especially below a 3:1 ratio of linear to full layers, and recommended 3:1 to 6:1 [R170].

Several large open-weight models now use this pattern:

- Qwen3-Next alternates three Gated DeltaNet layers with one gated-attention layer [R121]; the next generation repeats that layout at larger scale [R146].
- Kimi Linear and Kimi K3 repeat three Kimi Delta Attention layers followed by one latent-attention layer [R137] [R130]. Kimi Linear reports up to 75% less cache and up to 6× decoding throughput at one million tokens [R137], an author-reported figure.
- Nemotron-H and Nemotron 3 Super make most layers Mamba-2 and keep a small number of attention layers as global “anchors” [R171] [R126].

An illustrative calculation uses the published shape of Qwen3-Next: 48 layers, of which 12 are attention layers with 2 key/value heads of dimension 256 and 36 are Gated DeltaNet layers whose state per layer is 32 heads of `128 × 128` numbers. At 262,144 tokens and 2 bytes per value, the 12 attention layers cache 6.4 GB, while the 36 recurrent states total only 38 MB and do not grow with length. If all 48 layers were attention layers of that shape the cache would be 25.8 GB, four times larger (computed estimate, ignoring other small recurrent state and any higher-precision state storage). A recurrent layer’s state is as large as an attention layer’s cache for only about 512 tokens in this configuration (computed).

Hybrids are not universally adopted. MiniMax’s M2 model returned to full attention after the same lab had used a hybrid with linear attention earlier. Its engineers wrote that deficits appeared in complex multi-hop reasoning at larger scale, that evaluating such changes is expensive and benchmarks can mislead, and that the surrounding infrastructure was less mature for linear layers, naming low-precision state storage, prefix caching, and speculative decoding as problems still to solve [R154] [R172]. The sparse-attention route described earlier is an alternative way to cut long-context cost without a recurrent state.

::: warning Compare under controlled conditions
Claims that a hybrid, sparse, or recurrent design “beats” full attention depend on matched training tokens, data, parameters, kernels, and tasks. Quality at 8,000 tokens, recall at 1,000,000 tokens, and decoding speed can move in different directions.
:::

## Retrieval and memory architectures

Not all context must pass through dense attention. A system can retrieve relevant chunks from an external index, use recurrent memory tokens, compress old context, or call a database. This moves work between parametric memory, temporary context, and external storage.

No mechanism dominates every case:

- dense context preserves detail but is expensive and distractible;
- retrieval is fresh and attributable but can miss evidence;
- compressed memory is cheap but lossy;
- parameters are fast at inference but hard to update precisely.

A related architectural idea adds a lookup memory inside the model. **Engram** hashes recent n-grams to rows of a large embedding table, so retrieving a stored pattern is a constant-time lookup rather than something computed by layers; its authors frame it as a second axis of sparsity beside experts, and report a U-shaped trade-off when a fixed budget is split between the two [R173]. DeepSeek-V4.1-Flash includes 196 billion Engram parameters next to 552 billion backbone parameters [R152].

## Rewiring the residual stream

The residual connection `x + f(x)` has been stable for a decade. **Hyper-connections** replace the single stream by several parallel streams that learned matrices mix between layers [R174]. **Manifold-constrained hyper-connections** restrict the mixing matrix to doubly stochastic matrices, so the mapping cannot amplify signals, which restores stability that the unconstrained version lacked [R175]. DeepSeek-V4 expands its residual stream to four parallel streams this way [R139]. Kimi K3 instead lets each layer attend, with learned queries, over representations from earlier layers [R130]. Both are results reported by single labs, not settled defaults.

## Multimodal token streams

Images can be divided into patches and encoded as vectors; audio into frames or learned codes; video adds time; actions and sensor readings can also become token-like representations. A multimodal transformer may concatenate modalities, use cross-attention, or route through modality experts.

Generation can be autoregressive over discrete codes, diffusion-based in a continuous latent space, or hybrid. “One model” may actually contain separate encoders, decoders, diffusion components, and tool runtimes behind a shared interface.

## A dated snapshot of open architectures

The table lists architectures from technical reports and model cards checked for this edition. It is not a ranking and not complete. Report dates are first arXiv submissions, except for Qwen3-Next, where only the year is given.

| Report (date) | Attention layers | Feed-forward layers | Total / active parameters |
|---|---|---|---|
| DeepSeek-V3 (Dec 2024) | MLA in every layer | 1 shared + 256 routed experts, 8 active; first 3 layers dense | 671B / 37B |
| Kimi K2 (Jul 2025) | MLA | 1 shared + 384 routed experts, 8 active | 1.04T / 32.6B |
| gpt-oss-120b (Aug 2025) | Banded window (128) alternating with dense; GQA 64/8; sink bias | 128 experts, 4 active | 116.8B / 5.1B |
| Qwen3-Next-80B-A3B (2025) | 3 Gated DeltaNet : 1 gated attention (GQA 16/2) | 1 shared + 512 routed, 10 active | 80B / 3B |
| GLM-5 (Feb 2026) | MLA with DeepSeek Sparse Attention | 256 experts | 744B / 40B |
| Nemotron 3 Super (Apr 2026) | Mostly Mamba-2; few GQA attention layers (32/2), no positional embeddings | LatentMoE, 512 experts, 22 active | 120.6B / 12.7B |
| DeepSeek-V4-Pro (Apr 2026, preview) | Interleaved compressed sparse and heavily compressed attention, with window branch and sinks | 1 shared + 384 routed, 6 active | 1.6T / 49B |
| Gemma 4 31B (Jul 2026) | 5 local (1,024 window) : 1 global; p-RoPE and keys reused as values in global layers | Dense | 30.7B |
| Kimi K3 (Jul 2026) | 3 Kimi Delta Attention : 1 gated MLA, no positional encoding | LatentMoE, 2 shared + 896 routed, 16 active | 2.8T / 104B |
| DeepSeek-V4.1-Flash (Sep 2026) | Compressed sparse attention with cross-layer cache sharing; window in every layer; causal encoder–decoder | Shared + fine-grained routed experts; Engram memory | 552B backbone + 196B Engram / 8B (prefill), 16B (decode) |

Sources: [R66] [R93] [R120] [R121] [R129] [R126] [R139] [R138] [R147] [R130] [R152].

Three patterns stand out. Most of the large entries are sparse in their feed-forward layers; the one dense entry is a mid-sized model. All of the 2026 entries replace full attention in most layers or tokens with windows, compression, selection, or recurrence. And the choices disagree, which is the honest summary of the state of the field: grouped-query attention, latent attention, windows, sparse selection, and recurrent hybrids are all in use at the frontier of open weights.

# Predicting More Than One Token

The standard objective predicts only the next token at each position. Two lines of work change that. Part II (“Beyond one next token at a time”) introduces their training objectives; this chapter covers the architecture and decoding consequences.

## Multi-token prediction

**Multi-token prediction (MTP)** trains the model to predict several future tokens from each position. In the original formulation, `n` independent output heads sit on top of one shared trunk, and head `i` predicts the token `i` steps ahead; the extra losses are an auxiliary task [R118]. The authors report better sample efficiency, with larger gains on code and at larger model sizes, and that four-token models decode up to three times faster when the extra heads are used for speculative decoding [R118]. Speculative decoding drafts several tokens cheaply and verifies them in one pass of the full model [R34].

DeepSeek-V3 changed the design so that the extra predictions keep the causal chain [R66]. Its MTP module at depth `k` takes, for each position `i`, the hidden state from the previous depth and the embedding of token `i + k`, normalizes both, concatenates them (width `2d`), projects back to width `d`, and applies one transformer block. A shared output head turns the result into logits for token `i + k + 1`. Embedding and output head are shared with the main model. Depth-`k` modules see positions `1` to `T − k`. V3 used depth 1, a loss weight of 0.3 for most of training and 0.1 near the end, and could either drop the module at inference or reuse it as a speculative drafter; the authors report that the second token was accepted 85–90% of the time, giving 1.8 times the decoding speed [R66]. DeepSeek-V4 kept this configuration unchanged [R139]. GLM-5 shares parameters across three MTP layers during training, so that memory cost stays close to one layer while the draft acceptance length rises [R129]. Qwen3-Next, Qwen3.5, and Nemotron 3 Super also include MTP layers [R121] [R146] [R126].

Two uses must be kept apart. As a training signal, MTP densifies the learning signal and may push representations to plan ahead; the evidence is empirical and reported mostly by the authors. As a drafter, its value is speed: how many drafted tokens the full model accepts. DeepSeek-V4.1-Flash drops MTP from pretraining and instead trains a separate drafter after the backbone is frozen, called DSpark, which drafts five positions in one pass with a lightweight dependence between them and chooses how many to verify from confidence and server load [R152] [R176]. Which is better is not settled.

## Diffusion language models

An autoregressive model commits to one token at a time, left to right. A **diffusion language model** generates a whole sequence at once and refines it over several rounds.

In **masked discrete diffusion** the corruption process hides tokens: each token is replaced by a mask symbol independently with probability `t`, where `t` runs from 0 (nothing hidden) to 1 (everything hidden). A transformer without a causal mask sees the partly hidden sequence and predicts the hidden tokens, with cross-entropy computed on the hidden positions and weighted by `1/t`. That loss bounds the negative log-likelihood from above, which is what makes it a generative model, unlike BERT’s fixed masking ratio [R119]. A simplified form is a mixture of classical masked-language-model losses [R177].

To generate, start with the prompt followed by a fully masked response of a chosen length. In each of `N` steps the model predicts every masked position at once, keeps the most confident predictions, and re-masks the rest [R119]. Fewer steps are faster and lower quality [R119].

A toy count shows the trade. Generating 256 tokens autoregressively takes 256 sequential model calls, each processing one new position with a cache. With 64 diffusion steps over a 256-token response, there are 64 sequential calls, but each processes all 256 positions, so the model does 64 times as many position-computations in 4 times fewer sequential steps (computed). Decoding is often limited by reading weights from memory (see prefill and decode), and the extra positions share each weight read, so the extra arithmetic can be partly hidden. That is an argument from the structure of the cost, not a measured result.

Consequences:

- *Parallel output.* Many tokens can be finalized per call, which cuts latency when the sampler is well tuned.
- *Bidirectional context.* The model sees right-hand context as well as left, which helps infilling and editing.
- *No ordinary cache.* Because attention is bidirectional, a token’s representation changes when later tokens change, so the autoregressive cache is invalid. The first LLaDA model applied no cache and fixed the response length in advance as a user setting [R119]. **Block diffusion** generates blocks left to right and denoises within each block, so earlier blocks can be cached and the length is flexible [R178].
- *Likelihood.* Earlier diffusion models lagged autoregressive ones in likelihood modeling [R178]; masked diffusion with modern recipes was reported to approach autoregressive perplexity [R177].

Scale evidence is thinner than for autoregressive models. LLaDA trained an 8-billion-parameter model from scratch on 2.3 trillion tokens and reported in-context learning competitive with a similar-sized autoregressive model, while noting that its controlled baselines used less than `10²³` floating-point operations and that it had not scaled autoregressive baselines as far [R119]. LLaDA2.0 instead converts a pretrained autoregressive model to diffusion in stages, with mixture-of-experts variants of 16 and 100 billion parameters [R179]. A commercial line from Inception describes a coding model that, in measurements by an independent analyst, ran above 1,000 tokens per second on one GPU type [R180]; as of that company’s 19 October 2026 announcement, its latest diffusion model supports a one-million-token context [R181]. A joint model unifies autoregressive decoding, diffusion decoding, and a mode in which diffusion drafts and the autoregressive path verifies, and its authors report that the two objectives complement each other [R182].

Open questions: whether diffusion matches autoregressive quality at equal compute at frontier scale; how well reinforcement learning and long chains of reasoning transfer; how to serve bidirectional models efficiently with long contexts; and how to set the response length adaptively. The large open-weight models in the snapshot table above all generate text autoregressively.

# A Transformer Forward Pass in Pseudocode

The following omits batching details and numerical optimizations but preserves the conceptual graph:

```python
def attention(x, mask):
    q = x @ Wq
    k = x @ Wk
    v = x @ Wv
    q = apply_rotary(q)
    k = apply_rotary(k)
    # with GQA, k and v have fewer heads and each is shared by a group of query heads
    scores = (q @ transpose(k)) / sqrt(head_dim)
    scores = scores + mask
    weights = softmax(scores, axis="source_position")
    return concat_heads(weights @ v) @ Wo


def block(x, mask):
    x = x + attention(rms_norm(x), mask)
    x = x + gated_mlp(rms_norm(x))  # or a routed mixture of expert MLPs
    return x


def transformer(token_ids):
    x = embedding[token_ids]
    mask = causal_mask(length(token_ids))
    for layer in layers:
        x = layer.block(x, mask)
    x = rms_norm(x)
    return x @ vocabulary_projection
```

During training, compute logits for every position, compare with shifted targets, average token losses under the chosen weighting, and backpropagate.

## What is learned and what is fixed?

Learned:

- token and possibly position embeddings;
- attention projections, including latent down- and up-projections in MLA;
- MLP projections and gates;
- normalization scales;
- output projection;
- router parameters in MoE models;
- recurrent-layer gates and learned sink biases, where used.

Usually fixed by architecture or runtime:

- tensor shapes and number of layers;
- causal mask rule, window sizes, and local–global pattern;
- activation formula;
- tokenizer algorithm after vocabulary training;
- softmax operation;
- optimizer code, though its state evolves;
- in auxiliary-loss-free routing, the per-expert bias, which is adjusted by a load-counting rule rather than by gradients [R66].

Some recent work makes architecture choices more dynamic, but distinguishing parameters from algorithms remains useful.

## The complete sentence

A decoder-only transformer turns token IDs into embeddings, repeatedly lets each position route information from its allowed prefix and transform that information nonlinearly, projects the final context-dependent states into next-token logits, and learns all projections by reducing prediction loss over data.

If that sentence is clear at intuition, mechanism, shape, and equation levels, you understand the core of an LLM.

::: practice Paper reconstruction
Take a model’s architecture table and annotate: layers, model width, attention heads, key/value heads, head dimension, attention type per layer (full, windowed, sparse, latent, recurrent) and the pattern that repeats, MLP width, vocabulary, context, position method per layer type, norm, activation, dense or MoE, number of shared and routed experts and how many are active, multi-token-prediction depth, total parameters, active parameters. Then estimate weight memory and KV-cache memory, using the worked comparison above as a template and counting recurrent states separately from caches. This turns a brand name into an engineering object.
:::
