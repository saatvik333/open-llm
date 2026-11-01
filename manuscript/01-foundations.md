# Part I — Foundations

Every transformer operation reduces to ordinary computation on arrays of numbers. This part builds the minimum mathematics and computing intuition needed to see that machinery clearly. You can skim a familiar section and return when a later equation needs it.

# Computation: From Bits to Accelerators

A computer is a physical system whose states encode information and whose operations transform those states. Software creates layers of abstraction so that we can think in variables, functions, tensors, and models rather than individual transistor voltages.

## Bits, numbers, and representation

A **bit** has two states, written 0 and 1. Groups of bits encode integers, floating-point approximations, characters, images, instructions, and addresses. The representation is a convention. The bit pattern is not inherently the letter “A” or the number 3.5; a decoding rule gives it meaning.

With `n` bits, there are `2ⁿ` distinct patterns. Eight bits form a byte. Storage and memory are usually measured in multiples of bytes, though vendors and operating systems sometimes use different decimal and binary prefixes.

Neural networks mostly use floating-point numbers. A floating-point format stores a sign, a scaled exponent, and significant digits. It covers a wide range by sacrificing uniform precision. Narrower formats use less memory and bandwidth, but their encodings and computation need careful calibration. The next two sections lay out the formats in use as of this edition, from 32 bits down to 4, because the number format now shapes training stability, memory, and hardware speed.

Two consequences matter:

- computer arithmetic is approximate, so addition is not perfectly associative;
- numerical range and precision can determine whether training converges.

`(a + b) + c` can differ slightly from `a + (b + c)`. Summing millions of values, exponentiating logits (a network’s raw output scores, defined in Part II), or dividing by tiny norms requires stable algorithms.

## Floating-point formats for neural networks

A floating-point number has three fields: a **sign** bit, some **exponent** bits, and some **mantissa** bits (the stored fraction). For a normal number the value is:

`(−1)^sign × 2^(exponent − bias) × (1 + mantissa / 2^m)`

where `m` is the number of mantissa bits and `bias` is a fixed offset that lets an unsigned exponent field express negative powers. When the exponent field is all zeros, the format switches to **subnormal** numbers, which drop the implicit leading 1 so that values fade gradually toward zero instead of jumping to it [R70]. A format with `x` exponent bits and `y` mantissa bits is written `ExMy`.

The two field widths buy different things. Exponent bits set the **range**: how large or small a magnitude can be before it overflows or underflows. Mantissa bits set the **precision**: neighboring values differ by at most about `2^-y` of their size. Three mantissa bits leave a gap of 0.125 just above 1.0, so 1.05 and 1.00 become the same number.

| Format | Total bits | Sign / exponent / mantissa | Largest finite value | Note |
|---|---|---|---|---|
| FP32 | 32 | 1 / 8 / 23 | about 3.4 × 10³⁸ | master weights, optimizer state, accumulation |
| FP16 | 16 | 1 / 5 / 10 | 65,504 | more precise than BF16, narrow range |
| BF16 (bfloat16) | 16 | 1 / 8 / 7 | about 3.39 × 10³⁸ | same exponent range as FP32 |
| FP8 E4M3 | 8 | 1 / 4 / 3 | 448 | no infinities, a single NaN mantissa pattern |
| FP8 E5M2 | 8 | 1 / 5 / 2 | 57,344 | IEEE-style infinities and NaNs |
| FP6 (E3M2 or E2M3) | 6 | 1 / 3 / 2 or 1 / 2 / 3 | 28 or 7.5 | element type in block-scaled formats |
| FP4 E2M1 | 4 | 1 / 2 / 1 | 6 | element type in block-scaled formats |

The FP32, FP16, and BF16 layouts follow [R71]; the 8-, 6-, and 4-bit encodings follow [R70] and [R72].

**Sixteen bits.** FP16 has more precision than BF16 but a largest value of only 65,504, so small gradients underflow to zero and large activations overflow. Mixed-precision training works around this with **loss scaling** (multiply the loss by a constant before backpropagation so gradients stay in range, then undo it before the update) and by keeping a single-precision copy of the weights that receives the updates [R73]. Bfloat16 keeps FP32’s 8 exponent bits and gives up mantissa bits. Because its range matches FP32’s, training in it matched FP32 results across image, speech, language, and recommendation models without hyperparameter changes and without complicated loss scaling [R71].

The wider copy matters because of spacing. Just above 1.0, BF16 values are 2⁻⁷ ≈ 0.0078 apart, so adding an update of 0.003 to a weight of 1.0 returns exactly 1.0 and the update is lost. In the 4-bit training recipe described below, the master weights, weight-gradient accumulators, and optimizer state all stay in FP32 for this reason, even though most matrix multiplications run in 4 bits [R74].

**Eight bits.** The FP8 interchange proposal defines two encodings [R72]. E5M2 follows IEEE conventions, with infinities and several NaN patterns. E4M3 drops infinities and reserves a single mantissa pattern for NaN, which stretches its largest value to 448. The authors recommend E4M3 for weights and activations, where precision matters more, and E5M2 for gradients, which span a wider range. They matched 16-bit result quality on image and language tasks, including language models up to 175 billion parameters, with unchanged hyperparameters [R72]. By September 2025, a vendor report described 8-bit floating-point training as now widely adopted [R74]. The authors of DeepSeek-V3, an open-weight model reported in December 2024, state that no open-source large model had used FP8 for training before it; they used fine-grained scales, one per tile of 1 × 128 activation values and one per 128 × 128 block of weights [R75]. They later wrote that the matrix hardware they used summed FP8 products with only about 13 fractional bits, and asked future designs for FP32-class accumulation [R75].

**Scale factors and outliers.** Eight or four bits cannot span the range of real tensors, so a tensor is multiplied by a **scale factor** before conversion and divided by it afterward. One scale per tensor is cheap, but one huge value forces the scale down and rounds every ordinary value toward zero. Transformer activations contain such outliers; part of the purpose of SmoothQuant, a quantization method covered in Part VI, is to manage them [R44]. The remedy built into recent low-precision accelerators is to give each small block of values its own scale.

## Block scaling: microscaling and 4-bit formats

In the Open Compute Project’s **microscaling (MX)** standard, a block of `k` elements shares one 8-bit scale `X`, each element is stored in a narrow type `P`, and the value of element `i` is `X × P_i` [R70]. The scale is a power of two, stored as UE8M0 (unsigned, 8 exponent bits, no mantissa) with a range of 2⁻¹²⁷ to 2¹²⁷ [R70]. Version 1.0 of the standard appeared in September 2023; version 1.1, dated October 2026, renamed it “MicroXcaling” and renamed the scale type from E8M0 to UE8M0 [R70]. The 2023 paper behind it showed training language models with sub-8-bit weights, activations, and gradients [R76]. The standard defines four concrete formats, each with blocks of 32 elements [R70]. First-time readers can skim the rest of this section and return after the optimization chapter; later parts need only the idea that block scales are what keep 8- and 4-bit arithmetic usable.

| Format | Element type | Element bits | Bits per value, scale included |
|---|---|---|---|
| MXFP8 | FP8 (E5M2 or E4M3) | 8 | 8.25 |
| MXFP6 | FP6 (E3M2 or E2M3) | 6 | 6.25 |
| MXFP4 | FP4 (E2M1) | 4 | 4.25 |
| MXINT8 | 8-bit integer | 8 | 8.25 |

The last column is arithmetic: a block of 32 FP4 values plus an 8-bit scale costs `32 × 4 + 8 = 136` bits, or 4.25 bits per value. An E2M1 element can represent only eight non-negative values: 0, 0.5, 1, 1.5, 2, 3, 4, and 6 [R70, R74]. Everything else comes from the shared scale.

One conversion method the standard says implementations should support picks the power-of-two scale so that the block’s largest magnitude falls in the element type’s top power-of-two range, then rounds each scaled element to the nearest representable value, clamping anything above the maximum [R70]. This sketch applies it to one block of eight numbers (a real block has 32, and real hardware breaks ties to even, which the sketch ignores):

```python
import math

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]   # non-negative FP4 values; 6 = 1.5 * 2**2

def quantize_block(v):
    scale = 2.0 ** (math.floor(math.log2(max(abs(x) for x in v))) - 2)
    return scale, [math.copysign(min(E2M1, key=lambda g: abs(g - abs(x) / scale)), x) * scale
                   for x in v]

block = [0.9, -0.4, 0.05, 2.6, -1.1, 0.3, 0.0, 1.7]
print(quantize_block(block))
# (0.5, [1.0, -0.5, 0.0, 3.0, -1.0, 0.25, 0.0, 1.5])
```

The result shows what block scaling does and where it fails. Large values stay close (2.6 became 3.0, 1.7 became 1.5). The small value 0.05 underflowed to zero because it shares a scale with larger numbers. Smaller blocks isolate outliers better but spend more bits on scales.

::: practice Feed the block an outlier
Append `48.0` to the block and rerun the sketch. Before you look at the output, predict which of the original eight entries will become zero. Then explain why one outlier damages its neighbors, and why a per-tensor scale would be worse. Check: the scale jumps from 0.5 to 8, every original nonzero value except 2.6 rounds to zero, and 2.6 itself becomes 4.0.
:::

**A vendor variant.** NVFP4 keeps E2M1 elements but uses blocks of 16, stores each block’s scale in FP8 E4M3 (which has mantissa bits, so the scale need not be a power of two), and adds one FP32 scale per tensor [R74]. That costs `16 × 4 + 8 = 72` bits per block, or 4.5 bits per value, ignoring the single per-tensor scale. In that vendor’s 8-billion-parameter comparison, NVFP4 reached a lower training loss than MXFP4, and MXFP4 needed 36 percent more tokens to match it [R74]. Hardware with native block-scaled support reads the narrow elements together with their scales, computes each block’s partial dot product, applies the scales, and accumulates in FP32. The same vendor’s table lists matrix-unit math throughput at 2 times the 16-bit rate for 8-bit block formats and 4 to 6 times for 4-bit ones, depending on accelerator generation. These are rates for the arithmetic units alone, not delivered training speed [R74].

**Empirical finding: 4-bit pretraining can work, with care.** A 12-billion-parameter hybrid Mamba–Transformer (state-space and attention layers mixed; see Part III) trained on 10 trillion tokens in NVFP4 tracked an FP8 baseline closely: its validation loss stayed below 1 percent worse than the baseline during the stable phase and widened to slightly above 1.5 percent as the learning rate decayed [R74]. The recipe was not “round everything to 4 bits.” It kept about 16 percent of the linear layers, mostly near the end of the network, in higher precision; kept embeddings, the output head, normalization, and attention, including softmax, in BF16 or FP32; applied random Hadamard transforms (fixed orthogonal mixing matrices with random signs, which spread outliers over many values) to the inputs of weight-gradient multiplications; scaled weights in 16 × 16 two-dimensional blocks so that the forward and backward passes see the same quantized weights; and used **stochastic rounding** for gradients, which rounds a value up or down at random with probabilities set by its closeness to each neighbor so that rounding errors average out instead of biasing the gradient [R74].

**Open question: which 4-bit recipe?** Recipes are still moving. A 2026 controlled study of MXFP4 on an 8-billion-parameter model found that quantizing weight gradients caused most of the convergence damage, and that stochastic rounding and randomized Hadamard rotations failed to stabilize training once they were quantized, while deterministic Hadamard rotations did [R77]. A September 2026 paper reported stable software-emulated 4-bit pretraining of an 8-billion-parameter model for about 190 billion tokens using an unsigned E5M3 block scale, which has a wider range than E4M3, without Hadamard transforms [R78]. Both are small-scale results from single groups. Part IV covers how formats fit into full training recipes, and Part VI covers post-training quantization for inference.

## Programs, functions, and state

A **function** maps inputs to outputs. A pure function gives the same output for the same input and has no hidden side effects. Model layers are easiest to reason about as functions, even though real runtimes manage mutable buffers, random generators, caches, and devices.

An **algorithm** is a finite procedure. We often estimate its resource growth with input size:

- constant, `O(1)`;
- linear, `O(n)`;
- quadratic, `O(n²)`;
- logarithmic, `O(log n)`;
- exponential, such as `O(2ⁿ)`.

The notation hides constants and hardware behavior, but it explains why standard self-attention becomes expensive at long sequence lengths: the score matrix contains a relation for every pair of positions, so its size grows roughly with the square of sequence length.

## Memory hierarchy and data movement

Modern computation is limited not only by arithmetic. Data must move among storage, host memory, accelerator memory, caches, and arithmetic units. Nearby fast memory is small; distant large memory is slower.

A workload is **compute-bound** when arithmetic throughput limits speed and **memory-bound** when moving values limits speed. Generating one token for one user often reads a large fraction of model weights and can be memory-bandwidth-bound. Processing a large training batch reuses weights across many tokens and can better saturate matrix units.

This is why performance cannot be inferred from parameter count alone. Layout, precision, batch size, sequence length, kernel fusion, cache behavior, interconnects, and scheduling matter.

**Arithmetic intensity and the roofline.** A useful way to tell which limit applies is **arithmetic intensity**: the number of arithmetic operations performed per byte moved from memory. Divide a device’s peak arithmetic rate by its memory bandwidth and you get the intensity at which the two limits meet, the *ridge point*. A workload below it is memory-bound; above it, compute-bound. Plotting achievable speed against intensity gives the **roofline** model [R79]; a modern text works through the same analysis for transformer matrix multiplications and parallelism schemes [R80].

**Worked example (illustrative assumptions, not a product).** Take a dense model with 70 billion parameters stored as 8-bit weights, 70 GB in all, on an accelerator with 4 TB/s of memory bandwidth and 2,000 trillion 8-bit operations per second of peak matrix throughput. These are round numbers chosen for the arithmetic. Generating one token for one sequence must read every weight once: 70 GB ÷ 4 TB/s = 17.5 ms, so at most about 57 tokens per second for that sequence, however many arithmetic units sit idle. The ridge point is 2,000 ÷ 4 = 500 operations per byte. Each weight byte supports two operations (a multiply and an add) per sequence in the batch, so about 250 sequences must share each weight read before arithmetic, not memory, becomes the limit. Batching 64 sequences raises the weight-read-limited aggregate to about 3,660 tokens per second. The estimate ignores attention, the key–value cache (the stored keys and values of earlier tokens, covered in Part III), and scheduling.

**High-bandwidth memory.** Memory beside an accelerator’s arithmetic units is usually **HBM**: stacks of DRAM dies on the same package, read through a very wide interface. The JEDEC HBM4 standard, first published in April 2025 and revised in December 2025, specifies a 2,048-bit interface per stack at up to 8 Gb/s per pin, which is 2,048 × 8 ÷ 8 = 2,048 GB/s, about 2 TB/s per stack, with stacks of up to 16 dies and up to 64 GB each [R81]. An accelerator typically carries several stacks. Capacity still grows slower than demand: one published analysis puts growth of high-speed memory capacity below 50 percent per year, against growth of model memory demand above 1000 percent per year [R75]. This “memory wall” is a reason low-precision weights, compressed key–value caches, and sparse mixture-of-experts layers matter.

## CPUs, GPUs, and specialized accelerators

A CPU has a few sophisticated cores optimized for low-latency control flow and varied workloads. A GPU or tensor accelerator has many parallel arithmetic units optimized for applying similar operations across large arrays. Neural networks are dominated by matrix multiplications, which map well to these devices.

A **FLOP** is one floating-point operation. Hardware specifications may count a multiply and add as two FLOPs and quote peak rates in a particular precision. Peak FLOP/s are not delivered application throughput. Utilization accounts for idle units, communication, memory stalls, and unsupported shapes. Peak rates also depend on the number format: the same matrix unit can advertise several times more operations per second in 8-bit or 4-bit block-scaled formats than in 16-bit ones [R74].

Training compute for a dense decoder-only transformer is often estimated on the order of `6 × parameters × training tokens` FLOPs [R11]. This is a rough planning relation, not an exact law. The constant changes with architecture, sequence length, embedding treatment, recomputation, sparsity, and implementation. For a mixture-of-experts model, count the parameters active for each token, not the total. A published accounting lists about 250 GFLOPs per training token for a model with 671 billion total and 37 billion active parameters, and about 2,448 GFLOPs for a 405-billion-parameter dense model, at a sequence length of 4,096 [R75]. The rule gives `6 × 37 billion = 222` GFLOPs for the first, about 11 percent below the published figure, and `6 × 405 billion = 2,430` GFLOPs for the second, within 1 percent.

## Scale-up and scale-out networks

No single accelerator holds or trains a frontier model, so devices are joined by two kinds of network [R75]:

- A **scale-up** network links the accelerators within one node or rack. It offers the highest bandwidth and lowest latency, and can let one device read or write another’s memory directly with load and store operations. The UALink 200G 1.0 specification, released in April 2025 as an open standard, defines such a load/store link for up to 1,024 accelerators in one pod [R82].
- A **scale-out** network links nodes across a data center through network adapters and switches, typically using remote direct memory access (RDMA) over InfiniBand or Ethernet. It reaches far more devices but gives each far less bandwidth. The Ultra Ethernet 1.0 specification, released in June 2025, defines an open Ethernet-based communication stack for AI and high-performance computing [R83].

The gap between the two shapes model design. In one published open-weight training cluster built from a restricted-bandwidth accelerator variant, scale-up bandwidth was about four times scale-out bandwidth per device. Its designers avoided tensor parallelism during training because of the limited scale-up bandwidth, and restricted each token’s routed experts to at most four nodes so that cross-node traffic stayed bounded [R75]. The same authors estimated, for an idealized expert-parallel decoding step, that communication alone would cap speed at about 67 tokens per second per request stream over scale-out adapters, against about 1,200 over a 900 GB/s scale-up domain, assuming computation overlaps perfectly and ignoring network latency; they called the second figure purely theoretical [R75]. They also argue that future hardware should converge scale-up and scale-out into one framework [R75]. Part IV explains which parallelism strategies place which traffic on which network.

## Tensors and shapes

A **tensor** is a multidimensional array. In this book:

- a scalar has shape `[]`;
- a vector of model features has shape `[d_model]`;
- one token sequence has shape `[sequence, d_model]`;
- a batch has shape `[batch, sequence, d_model]`.

Shape annotations are executable intuition. For a matrix multiplication `[m, k] @ [k, n] → [m, n]`, the inner dimension must match. In a model, the names carry semantics: batch, position, head, feature, vocabulary.

::: practice Shape drill
Suppose a batch has 8 sequences, each 512 tokens, and each token is represented by 1,024 features. The activation tensor shape is `[8, 512, 1024]`. In FP16 it stores `8 × 512 × 1024 × 2 ≈ 8.4` MB before allocator overhead; in an 8-bit format it needs half that, about 4.2 MB, plus a small allowance for scale factors. Training retains many such activations across layers, which is why activation memory can dominate.
:::

# Linear Algebra: The Geometry of Models

Linear algebra describes spaces of features and transformations between them. A neural network’s learned knowledge is distributed across many matrices; almost every layer is built around matrix multiplication.

## Scalars, vectors, and matrices

A scalar is one number. A vector is an ordered list such as:

`x = [2, -1, 0.5]`

A matrix is a rectangular table. Multiplying a vector by a matrix mixes its components. If **x** has `d_in` components and **W** has shape `[d_in, d_out]`, then:

`y = xW`

has `d_out` components. A bias vector **b** adds an offset:

`y = xW + b`

This is an **affine transformation**. Neural layers learn **W** and **b**.

## The dot product

The dot product multiplies corresponding components and sums:

`x · y = Σᵢ xᵢ yᵢ`

It has three useful interpretations:

- a weighted sum;
- a measure of alignment;
- the projection of one vector along another, scaled by lengths.

If vectors are normalized to length one, their dot product is cosine similarity. Attention scores and embedding retrieval often use dot products because accelerators compute them efficiently and training can organize useful geometry around them.

A high dot product is not a universal semantic law. Its meaning comes from how the vectors were trained and normalized.

## Bases and learned coordinates

Coordinates depend on a basis. The same physical arrow can have different coordinates under rotated axes. Likewise, individual neural features are usually not intrinsically meaningful. A representation can be transformed and later untransformed without changing the network’s function.

This creates an interpretability warning: attaching a human label to one neuron may be unstable. Meaning can be **distributed** across dimensions or **superposed**, with a limited set of neurons representing more features than there are dimensions.

## Matrix multiplication as parallel feature construction

For a batch matrix **X** of shape `[tokens, d_in]` and a weight matrix **W** of shape `[d_in, d_out]`, **XW** computes `d_out` new features for every token in parallel. Each output column is a learned weighted combination of input features.

Composing only linear transformations remains linear:

`(xW₁)W₂ = x(W₁W₂)`

Depth becomes useful when nonlinear operations occur between transformations. That is why activations such as ReLU, GELU, sigmoid, or gated variants matter.

## Norms and distance

A norm measures vector size. The Euclidean or L2 norm is:

`||x||₂ = sqrt(Σᵢ xᵢ²)`

The L1 norm sums absolute values. Norms appear in regularization, normalization, optimization, similarity, clipping, and error metrics.

High-dimensional geometry can violate low-dimensional intuition. Random vectors tend to be nearly orthogonal; distances can concentrate; a small change in each of many dimensions can add up. Empirical validation is safer than visual metaphors.

## Eigenvectors, singular values, and rank

An eigenvector keeps its direction under a square linear transformation. Singular value decomposition generalizes useful structure to any matrix:

`W = UΣVᵀ`

The singular values in **Σ** describe strengths of orthogonal transformation directions. If only a few are important, a low-rank approximation can preserve much of the behavior with fewer numbers.

Low-rank adaptation (LoRA) exploits this idea operationally. Instead of updating a full weight matrix **W**, it learns a low-rank change:

`W' = W + BA`

where **A** and **B** are narrow matrices. This greatly reduces trainable parameters and optimizer state, though the best rank and target modules are empirical choices [R35].

# Probability: Reasoning Under Uncertainty

A language model produces a probability distribution, not a sentence directly. Decoding turns distributions into a particular sequence.

## Events and distributions

A probability lies between 0 and 1. Mutually exclusive outcomes in a complete discrete distribution sum to 1. If the next-token vocabulary contains four tokens, a prediction might be:

| Token | Probability |
|---|---|
| `red` | 0.55 |
| `blue` | 0.25 |
| `green` | 0.15 |
| `</s>` | 0.05 |

These values express model uncertainty under a context. They are not automatically calibrated frequencies about factual correctness.

A **random variable** maps outcomes to numbers. Its expected value is a probability-weighted average:

`E[X] = Σₓ p(x)x`

Variance measures expected squared deviation from the mean. Covariance measures how two variables vary together. Correlation is normalized covariance, but neither correlation nor model attention alone proves causation.

## Conditional probability and the chain rule

Conditional probability asks about `A` given `B`:

`P(A | B) = P(A and B) / P(B)`

A sequence probability factors by the probability chain rule:

`P(x₁, x₂, …, x_T) = Πₜ P(x_t | x₁, …, x_{t-1})`

This identity is the foundation of autoregressive language modeling. The model estimates each conditional factor; multiplying them gives a probability for the complete sequence. Because products of many small numbers underflow, implementations add log-probabilities instead.

Bayes’ rule reverses a condition:

`P(H | E) = P(E | H)P(H) / P(E)`

A prior `P(H)` is updated by evidence likelihood `P(E | H)` into posterior `P(H | E)`. Bayesian language is useful for thinking, but a neural model’s token probabilities should not be casually interpreted as a complete Bayesian posterior over real-world hypotheses.

## Independence and dependence

Two variables are independent if learning one gives no information about the other. Conditional independence can hold after accounting for another variable. Graphical models use such structure explicitly.

Language is densely dependent. The right interpretation of a pronoun may depend on words far earlier; code indentation depends on syntax; an answer depends on instructions and retrieved evidence. Model architecture and context determine which dependencies can be represented and used.

## Likelihood

For observed data **D** and parameters θ, the likelihood is `P(D | θ)` viewed as a function of θ. Maximum-likelihood training chooses parameters that make observed examples probable. For autoregressive language modeling, maximizing sequence likelihood is equivalent to minimizing negative log-likelihood over target tokens.

This objective rewards matching the training distribution. It does not by itself mean “be truthful,” “follow the user,” “be harmless,” or “admit uncertainty.” Those behaviors need data, objectives, architecture, prompting, verification, or policy that support them.

# Information Theory: Surprise, Entropy, and Loss

Information theory connects probability to compression. A good predictor can encode common events with short codes and surprising events with longer codes.

## Self-information and entropy

The surprise of an event with probability `p` is:

`I(x) = -log p(x)`

A likely event carries less surprise. The log base determines units: base 2 gives bits; natural log gives nats.

Entropy is expected surprise:

`H(P) = -Σₓ P(x) log P(x)`

A distribution concentrated on one certain outcome has zero entropy. A uniform distribution over many outcomes has high entropy.

Entropy describes uncertainty under a distribution. It is not disorder in every colloquial sense, and high token entropy does not directly measure philosophical creativity.

## Cross-entropy

Suppose the data follow distribution **P** and the model predicts **Q**. Cross-entropy is:

`H(P, Q) = -Σₓ P(x) log Q(x)`

For one training target represented as a one-hot distribution, this reduces to negative log-probability of the correct token. If the model gives the target probability 0.8, loss is `-log(0.8)`; if it gives 0.001, the loss is much larger.

Averaging this across tokens gives the standard next-token training loss. Lower loss means better probabilistic compression on that evaluation distribution. It does not guarantee a better assistant on every downstream objective.

## KL divergence

Kullback–Leibler divergence measures extra expected code length from using **Q** when **P** is true:

`KL(P || Q) = Σₓ P(x) log(P(x)/Q(x))`

It is nonnegative and zero when the distributions match, but it is not symmetric and therefore not a distance metric. KL terms appear in distillation, where they pull a student toward a teacher, and in preference optimization, where a KL penalty keeps a policy near a reference model.

The relation is:

`H(P, Q) = H(P) + KL(P || Q)`

Because data entropy `H(P)` does not depend on model parameters, minimizing cross-entropy also minimizes the forward KL from data to model in the idealized distribution view.

## Perplexity

Perplexity is the exponential of average cross-entropy measured in nats (with a loss in bits, use `2^loss` instead):

`perplexity = exp(loss)`

Roughly, it is the effective branching factor under the model. A perplexity of 10 suggests uncertainty comparable to choosing among ten equally likely options at each step, although real distributions are not uniform.

Never compare perplexity values across different tokenizers without care. Splitting the same text into different units changes the number and difficulty of predictions. A tokenizer-independent alternative is **bits per byte**: the total loss in bits divided by the number of bytes of text [R84]. Part II works an example.

::: idea Compression lens
Learning regularities lets a model compress data. Useful syntax, facts, styles, and algorithms can all improve prediction. Compression explains why next-token training can acquire broad structure—but not why every desired behavior must emerge.
:::

# Calculus and Automatic Differentiation

Training asks: how should millions or trillions of parameters change to reduce loss? Calculus provides local sensitivity; backpropagation organizes it efficiently.

## Derivatives as sensitivity

For a scalar function `y = f(x)`, the derivative `dy/dx` describes how a small change in `x` changes `y`:

`f(x + Δx) ≈ f(x) + (df/dx)Δx`

If the derivative is positive, increasing `x` locally increases `f`. If its magnitude is large, the output is sensitive.

A partial derivative changes one input while holding others fixed. For loss `L(w₁, w₂, …)`, the gradient is the vector of all partial derivatives:

`∇L = [∂L/∂w₁, ∂L/∂w₂, …]`

The gradient points toward steepest local increase. Moving a small step in the negative gradient direction tends to decrease loss.

## The chain rule

Neural networks compose functions. If `y = f(u)` and `u = g(x)`, then:

`dy/dx = (dy/du)(du/dx)`

The chain rule passes sensitivity backward through each operation. For a long computation graph, the same intermediate value may influence loss through several paths; its gradient is the sum of those contributions.

## Forward and reverse mode

Automatic differentiation records primitive operations and applies exact derivative rules to the executed computation. It is not symbolic algebra and not finite-difference approximation.

- **Forward mode** propagates how outputs change with one input direction.
- **Reverse mode** starts from a scalar output loss and propagates sensitivities to many inputs.

Because a model has many parameters and usually one scalar loss, reverse mode is efficient. Backpropagation is reverse-mode automatic differentiation specialized to layered computations.

## A scalar example

Let:

`a = wx + b`

`y = a²`

For a target `t`, let `L = (y - t)²`. Reverse differentiation proceeds:

- `∂L/∂y = 2(y - t)`
- `∂y/∂a = 2a`
- `∂a/∂w = x`
- therefore `∂L/∂w = 2(y - t) × 2a × x`

A framework performs this bookkeeping for tensor operations. Understanding the scalar case makes the tensor case less mysterious.

## Vanishing and exploding gradients

Repeated multiplication by derivatives smaller than one can shrink a signal toward zero; factors larger than one can make it explode. Residual connections, normalization, careful initialization, gated activations, gradient clipping, and architecture design help, though finite gradients can still be unhelpful under saturation, poor conditioning, noisy objectives, or sharp curvature. **Gradient clipping** rescales the whole gradient when its global norm exceeds a threshold, guarding against rare huge steps; one published large-model recipe clipped to a global norm of 1.0 [R85].

# Optimization: Turning Gradients into Learning

A gradient is a local direction, not a complete route through a complicated landscape. An optimizer combines gradients over many examples and steps to find parameters with low loss.

## Gradient descent and stochastic gradients

Full-batch gradient descent computes the average gradient over the complete dataset:

`θ ← θ - η∇L(θ)`

where θ contains parameters and η is the learning rate. For enormous datasets this is impractical. **Stochastic gradient descent** estimates the gradient from a mini-batch. The estimate is noisy but cheap, and the noise can help exploration.

One **step** processes one mini-batch and updates parameters. An **epoch** processes roughly every training example once. Foundation-model pretraining may describe progress in tokens rather than epochs because the data mixture is huge and may not have a simple pass structure.

## Learning rate

Too large a learning rate can diverge; too small wastes compute or settles poorly. Schedules often warm up from a small value, then decay. The effective step also depends on batch size, optimizer state, normalization, clipping, and precision.

A larger batch reduces gradient noise and increases parallelism, but beyond a regime it gives diminishing returns unless other hyperparameters change. Gradient accumulation emulates a larger batch by summing gradients across micro-batches before updating.

A schedule common in language-model pretraining is **warmup–stable–decay (WSD)**: raise the rate from near zero, hold it constant for most of training, then decay it over a short final phase [R86]. One 12-billion-parameter run held the rate constant for the first 80 percent of training and decayed it over the last 20 percent [R74]. Because the stable phase has no preset end, training can be continued or adapted from a mid-run checkpoint, which is why the schedule suits continued training and domain adaptation [R86].

Good learning rates also depend on model width. In the maximal update parametrization (μP), the best learning rate and other hyperparameters stay stable as width grows, so values tuned on a small model transfer to a much larger one without re-tuning [R87]. Transfer was demonstrated empirically on transformers and residual networks, and the authors note caveats for other axes such as depth; it is a recipe, not a guarantee.

## Momentum, Adam, and AdamW

Momentum smooths gradients with a moving average, building velocity in consistent directions. Adam maintains moving estimates of the first and second moments of gradients, scaling updates per parameter. AdamW decouples weight decay from the adaptive gradient update and is common in transformer training [R16].

An intuitive Adam-like sketch is:

```text
m = β1*m + (1-β1)*gradient
v = β2*v + (1-β2)*gradient²
parameter -= learning_rate * corrected(m) / (sqrt(corrected(v)) + epsilon)
```

The exact details—bias correction, precision, clipping, distributed reduction—matter in real implementations.

## Beyond AdamW: matrix-aware optimizers

AdamW rescales each parameter using only that parameter’s own gradient history. But most transformer parameters sit in weight matrices, and a matrix acts through its singular values, as the linear-algebra chapter described. The update that SGD with momentum or Adam proposes for a weight matrix tends to be dominated by a few directions, with the rest barely moving; the author of one widely used alternative describes such updates as having a very high condition number and speculates that orthogonalizing them raises the scale of rare but important directions [R88].

**Muon** (MomentUm Orthogonalized by Newton–Schulz) targets this. For each hidden-layer weight matrix it keeps a momentum average of gradients, replaces that momentum matrix with the nearest semi-orthogonal matrix, and applies the result with a learning rate [R88]. Replacing means keeping the singular vectors and setting every singular value to one: if the SVD is `UΣVᵀ`, the update becomes `UVᵀ`. A full SVD each step would be too slow, so Muon approximates the result with a few steps of the Newton–Schulz iteration, usually five, which uses only matrix multiplications and runs stably in bfloat16 [R88]. In the original recipe, embeddings, the output layer, and scalar or vector parameters such as normalization gains are still trained with AdamW [R88]. The author estimates the extra arithmetic at under 1 percent of training FLOPs for typical language-model runs [R88].

This sketch shows the effect on a random matrix whose singular values are deliberately uneven (the coefficients are those published with Muon; real implementations run in bfloat16):

```python
import numpy as np

rng = np.random.default_rng(0)
G = rng.standard_normal((256, 64)) @ np.diag(np.linspace(5.0, 0.05, 64))

def spread(M):                       # largest and smallest singular value
    s = np.linalg.svd(M, compute_uv=False)
    return round(float(s.max()), 2), round(float(s.min()), 2)

U, S, Vt = np.linalg.svd(G, full_matrices=False)
print("update:", spread(G))          # (90.97, 0.67)
print("exact: ", spread(U @ Vt))     # (1.0, 1.0)

def newton_schulz(G, steps=5, a=3.4445, b=-4.7750, c=2.0315):
    X = G / (np.linalg.norm(G) + 1e-7)     # scale so singular values are at most 1
    if X.shape[0] > X.shape[1]:
        X = X.T
    for _ in range(steps):                 # matrix multiplies only
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if G.shape[0] > G.shape[1] else X

print("approx:", spread(newton_schulz(G)))   # (1.18, 0.68)
```

The raw update’s singular values span a factor of about 136; after orthogonalization they are all 1, or within roughly 0.7 to 1.2 after five cheap steps.

**Shampoo and relatives.** Muon is closely related to Shampoo, which multiplies the gradient on both sides by matrices built from gradient statistics [R89]. With the statistics accumulation removed, Shampoo’s update equals the orthogonalized gradient `UVᵀ`, which is how Muon’s author connects the two [R88]. Distributed Shampoo won the external-tuning ruleset of the inaugural AlgoPerf training-algorithms competition, and a schedule-free variant of AdamW won the self-tuning, hyperparameter-free ruleset [R90]. SOAP runs Adam in the eigenbasis of Shampoo’s preconditioner [R91].

**Evidence, with caveats.**

- Early evidence came from speed-record competitions on small language models, where switching from AdamW to Muon set new records [R88].
- A 2025 study from Moonshot AI found that two changes let Muon scale without retuning, adding weight decay and adjusting each parameter’s update scale to match AdamW’s, and reported about twice the compute efficiency of AdamW in compute-optimal scaling experiments [R92]. That is the authors’ own comparison.
- Moonshot AI’s Kimi K2, a mixture-of-experts model with 1 trillion total parameters, was pretrained on 15.5 trillion tokens with a Muon variant, MuonClip, and reported no loss spikes. MuonClip adds QK-Clip, which rescales query and key weights after each update to bound attention logits (see Part II, “Keeping large training runs stable”) [R93].
- Independent tuning-controlled comparisons are more sober. A 2025 study that tuned ten optimizers across 0.1 to 1.2 billion parameters found matrix-based optimizers such as Muon and SOAP fastest, but their speedup over a tuned AdamW shrank from about 1.4 times at 0.1 billion parameters to about 1.1 times at 1.2 billion, and rankings could flip as the learning rate decayed [R94].
- A July 2026 study by NVIDIA researchers, covering multi-billion-parameter models on trillions of tokens, found SOAP and Muon outperforming AdamW at the scales tested, and stable at batch sizes up to 100 million tokens where AdamW degraded [R95].

**Judgment, as of this edition.** AdamW remains the baseline against which alternatives are judged. Matrix-aware optimizers have a trillion-parameter-scale report and a multi-billion-parameter vendor study behind them, yet carefully tuned small-scale comparisons show gains smaller than first claimed, and results depend on tuning, scale, batch size, and schedule. When you read an optimizer claim, ask whether the AdamW baseline was tuned as carefully as the new method, and whether the comparison was made at the end of training rather than at an intermediate checkpoint [R94].

## Regularization and generalization

Optimization minimizes the measured objective; **generalization** asks whether performance transfers to unseen examples. Useful tools include:

- weight decay;
- dropout during training;
- data augmentation and diverse mixtures;
- early stopping;
- capacity and architecture choices;
- label smoothing in some tasks;
- validation-based selection.

Large language models complicate the classic picture. They can memorize rare strings while also learning transferable algorithms. Dataset contamination makes “unseen” hard to establish. A model can generalize compositionally in one domain and fail under a small distribution shift in another.

## Local minima are not the whole story

Deep-network loss landscapes are high-dimensional. Saddle points, flat and sharp directions, optimization path, parameter symmetries, and implicit bias matter. “The optimizer found the global minimum” is usually neither known nor necessary. Many parameter settings can achieve similar training loss but differ in generalization and behavior.

# Learning from Data

Machine learning connects an objective, a hypothesis class, data, and an optimization algorithm. None can be judged in isolation.

## Major learning settings

**Supervised learning** observes inputs paired with desired labels or outputs. Spam detection and image classification are common examples.

**Unsupervised learning** seeks patterns without explicit task labels: clustering, density estimation, latent structure.

**Self-supervised learning** constructs targets from the data. Predicting masked or next tokens uses the sequence itself as supervision. This is how enormous unlabeled corpora become training examples.

**Reinforcement learning** learns behavior from rewards in an environment. Actions influence future observations, so credit assignment spans time. In LLM post-training, the “environment” may be a prompt plus verifier, user simulator, code executor, or human preference process.

**Online learning** updates as data arrive; **offline learning** trains on a fixed dataset. Deployed assistants may collect feedback, but responsible systems do not silently train on every conversation without controls.

## Train, validation, and test

A training set fits parameters. A validation or development set selects hyperparameters and checkpoints. A test set estimates final performance only after choices are fixed.

If you repeatedly tune on the test set, it becomes another validation set. Public benchmarks are especially vulnerable because their questions, solutions, or close variants can enter training data and because the research community collectively tunes against them.

Use time splits for future prediction, group splits when examples share entities, and contamination audits for web-trained models. Match deployment conditions: a clean multiple-choice benchmark is not a substitute for noisy conversational traffic.

## Bias, variance, and irreducible uncertainty

A simple model may underfit systematic structure: high bias. An excessively flexible model may fit noise or dataset peculiarities: high variance. More and better data, regularization, ensembles, and inductive biases change the trade-off.

Some uncertainty is **aleatoric**: ambiguity or randomness in the data. Some is **epistemic**: uncertainty from limited knowledge or model specification. The distinction is useful but not always identifiable from one model’s token probabilities.

## Distribution shift

Training assumes examples represent future use. Reality violates this:

- **covariate shift:** inputs change;
- **label or concept shift:** relationships or targets change;
- **domain shift:** language, population, tools, or institutions differ;
- **adversarial shift:** someone deliberately finds failures;
- **feedback shift:** the deployed model changes the data people create.

LLMs face all of these. Evaluation should include slices, stress tests, red teaming, and post-deployment monitoring, not one average score.

## Data quality is objective design

A dataset does not merely feed the objective; its composition partially defines the objective. If 40% of tokens come from code, code patterns receive roughly that opportunity to influence gradients, modified by sampling and loss weighting. Deduplication, language balance, source quality, chronology, formatting, and filtering shape behavior.

“More data” can add misinformation, private information, duplicated boilerplate, benchmark answers, toxic material, or machine-generated degradation. “Clean data” is not neutral either: filters embed definitions of quality and may erase dialects, minority knowledge, or difficult but important content.

Datasheets [R51] and data statements document origin, collection, intended use, consent, preprocessing, limitations, and maintenance. Model cards do a related job for trained models [R50]. Documentation is not proof of safety, but missing documentation makes scrutiny harder.

## Causation versus prediction

A predictor can exploit correlation without learning the causal mechanism. If hospital type correlates with illness severity, a model may use hospital identity as a shortcut. Intervening on the world can break that correlation.

Causal questions require assumptions and designs such as randomized experiments, natural experiments, causal graphs, or domain knowledge. A language model can help formulate hypotheses or analyze evidence, but fluent causal language does not create identification.

::: practice Data audit
For any model claim, write down: target population, sampling process, label process, time period, unit of analysis, possible duplicates, missing groups, deployment shift, and who bears an error. This often reveals more than another decimal place of benchmark performance.
:::
