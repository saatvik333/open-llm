# Appendix A — Formula and Shape Sheet

## Probability and information

| Concept | Formula | Meaning |
|---|---|---|
| Conditional probability | `P(A|B)=P(A,B)/P(B)` | probability of A after observing B |
| Bayes’ rule | `P(H|E)=P(E|H)P(H)/P(E)` | update hypothesis with evidence |
| Sequence factorization | `P(x₁…x_T)=ΠₜP(x_t|x_<t)` | autoregressive foundation |
| Self-information | `I(x)=-log p(x)` | surprise of an outcome |
| Entropy | `H(P)=-ΣP(x)log P(x)` | expected uncertainty |
| Cross-entropy | `H(P,Q)=-ΣP(x)log Q(x)` | expected predictive loss |
| KL divergence | `KL(P||Q)=ΣP log(P/Q)` | mismatch in one direction |
| Perplexity | `exp(average cross-entropy in nats)` | effective branching factor |
| Softmax with temperature | `softmax(z/T)_i = exp(z_i/T) / Σ_j exp(z_j/T)` | logits to probabilities; `T<1` sharpens, `T>1` flattens |
| Bits per byte | `(loss in nats per token) / (ln 2 × bytes per token)` | loss comparable across tokenizers |

## Neural computation

| Concept | Formula | Typical shapes |
|---|---|---|
| Affine map | `Y=XW+b` | `[B,d_in]@[d_in,d_out]→[B,d_out]` |
| MLP | `Y=φ(XW₁+b₁)W₂+b₂` | width expands then contracts |
| Residual | `Y=X+F(X)` | X and F(X) match |
| Gradient update | `θ←θ-η∇L` | one value per parameter |
| LoRA | `W'=W+BA` | rank of BA is at most r |
| RMSNorm | `g⊙x/sqrt(mean(x²)+ε)` | normalizes feature scale |
| Gated MLP (SwiGLU) | `(SiLU(xW_gate)⊙xW_up)W_down` | two expanding projections, one contracting |
| Muon update | `UVᵀ`, where the momentum matrix is `UΣVᵀ` | every singular value set to one; a few Newton–Schulz iterations approximate this |

## Number formats and hardware limits

| Concept | Formula | Meaning |
|---|---|---|
| Floating-point value | `(−1)^sign × 2^(exponent−bias) × (1+mantissa/2^m)` | normal numbers with `m` mantissa bits; `ExMy` has `x` exponent and `y` mantissa bits |
| Block-scaled element | `value_i = X × P_i` | one 8-bit scale `X` shared by a block of `k` narrow elements `P_i` |
| Bits per value, block-scaled | `(k × element bits + scale bits) / k` | MXFP4: `(32×4+8)/32 = 4.25`; NVFP4: `(16×4+8)/16 = 4.5`, ignoring the per-tensor scale |
| Ridge point | `peak operations per second / memory bandwidth` | operations per byte; work below it is memory-bound |
| Single-sequence decode bound | `bytes read per token / memory bandwidth` | minimum seconds per token; 70 GB at 4 TB/s is 17.5 ms |

## Training objectives

| Concept | Formula | Meaning |
|---|---|---|
| Contrastive loss (InfoNCE) | `−log( exp(s⁺/τ) / Σ_j exp(s_j/τ) )` | softmax over the positive and all negatives; `τ` is the temperature |
| z-loss | `10⁻⁴ × (log Z)²`, with `Z = Σ_i exp(z_i)` | penalizes a large softmax denominator; adding a constant to every logit leaves probabilities unchanged but shifts `log Z`, so the penalty changes, and it is zero only at `log Z = 0` |
| Multi-token prediction | sum of cross-entropies for tokens `t+1 … t+n` | `n` heads on one shared trunk (original formulation) |
| Masked-diffusion loss | cross-entropy on masked positions, divided by `t` | mask ratio `t` in `(0,1]`; upper-bounds the negative log-likelihood |
| Scaling-law form | `L(N,D) ≈ L∞ + aN^{-α} + bD^{-β}` | empirical fit over a measured regime, not a promise |

## Reinforcement learning with verifiable rewards

Group relative policy optimization (GRPO), as written in Part IV:

```text
sample o_1 … o_G from π_old(· | q);  reward r_i for each response o_i
A_i    = (r_i − mean(r_1 … r_G)) / std(r_1 … r_G)
ρ_i,t  = π_θ(o_i,t | q, o_i,<t) / π_old(o_i,t | q, o_i,<t)
J(θ)   = average over i of  (1 / |o_i|) · sum over t of
           [ min( ρ_i,t · A_i ,  clip(ρ_i,t, 1−ε, 1+ε) · A_i )  −  β · KL_i,t ]
KL_i,t = π_ref / π_θ − log(π_ref / π_θ) − 1      (both probabilities at token o_i,t)
maximize J over θ
```

- If every reward in a group is equal, the standard deviation is zero and the prompt gives no learning signal.
- GSPO replaces the token ratio with a sequence ratio: `(π_θ(o | q) / π_old(o | q))^(1 / |o|)`.
- A length-controlled reward (L1-Exact): `reward = (1 if correct else 0) − α · |n_target − n_generated|`.

## Attention

For batch `B`, sequence `T`, heads `H`, head width `D`:

- Q, K, V: `[B,H,T,D]`
- scores `QKᵀ`: `[B,H,T,T]`
- attention weights: `[B,H,T,T]`
- per-head output: `[B,H,T,D]`

`Attention(Q,K,V)=softmax(QKᵀ/√D + mask)V`

Causal mask allows source index `j ≤ i` for destination `i`.

## Attention variants, position, and recurrent state

| Concept | Formula | Meaning |
|---|---|---|
| Grouped-query attention | Q `[B,H,T,D]`; K, V `[B,G,T,D]` | `H/G` query heads share each key/value head; `G=1` is multi-query, `G=H` is multi-head |
| Multi-head latent attention | `c_t = W_DKV x_t`, `k_t = W_UK c_t`, `v_t = W_UV c_t` | cache only the small latent `c_t` plus a decoupled rotary key |
| Attention with a sink bias | `a_j = exp(z_j) / (exp(b) + Σ_k exp(z_k))` | learned `b` per head; the weights can sum to less than 1 |
| RoPE frequency and wavelength | `f_j = 10000^(−2j/D)`, wavelength `2π/f_j` | pair `j = 0 … D/2−1`, original formulation |
| Position interpolation | position `m` becomes `m/s` | extension factor `s` keeps rotary angles within the trained range |
| Linear attention | `S ← S + v kᵀ`, `o = S q` | fixed-size state `[d_v,d_k]` instead of a growing cache; the published version adds feature maps and a normalizer |
| Decay (gating) | `S ← α S + v kᵀ`, with `0<α<1` | uniformly forgets old content |
| Delta rule | `S ← S (I − β k kᵀ) + β v kᵀ` | overwrites the association for a unit-norm key `k` |
| Gated delta rule | `S ← S · α (I − β k kᵀ) + β v kᵀ` | decay plus overwrite |

## Embeddings and retrieval

| Concept | Formula | Meaning |
|---|---|---|
| Matryoshka truncation | keep the first `m` of `d` values, then renormalize | the prefix works as an embedding; cosine similarity needs unit length |
| Late-interaction score | `Σ_q max_d sim(q, d)` | each query token’s best document-token match, summed over query tokens |

## Mixture of experts

| Concept | Formula | Meaning |
|---|---|---|
| Active fraction | `active parameters / total parameters` | DeepSeek-V3: `37B / 671B ≈ 5.5%` |
| Forward cost per token | `≈ 2 × active parameters` FLOPs | ignores attention over the context |
| Experts touched by a batch | `E × (1 − (1 − k/E)^B)` | `E` experts, `k` active per token, `B` tokens, independent uniform routing; `E=384, k=8, B=64` gives about 284 |
| Auxiliary-loss-free routing | choose the top `k` of `affinity + bias`; mix with weights from `affinity` alone | bias is lowered for overloaded experts and raised for underloaded ones, not trained by gradient |

## Decoding and evaluation

| Concept | Formula | Meaning |
|---|---|---|
| Tokens per verification pass (speculative decoding) | `(1 − α^(γ+1)) / (1 − α)` | `α` acceptance probability per drafted token, `γ` drafted tokens; independent acceptance assumed |
| pass@k | `1 − (1 − p)^k` | at least one of `k` independent trials of a task succeeds, per-trial success probability `p`; from `n ≥ k` samples with `c` correct, the unbiased estimate is `1 − C(n−c,k) / C(n,k)`; benchmarks average over tasks |
| pass^k | `p^k` | all `k` independent trials of a task succeed; averaged over tasks |

## Memory approximations

- Weight bytes: `parameters × bits_per_parameter / 8`; for block-scaled formats include the scales in the bits per parameter.
- Dense training compute: roughly `6 × parameters × training_tokens` FLOPs, recipe-dependent; for a mixture-of-experts model use the parameters active for each token.
- Lifetime compute: `6·N·D_train + 2·N·D_served` FLOPs for `N` parameters, `D_train` training tokens, and `D_served` tokens processed in serving; arithmetic only.
- Compute-optimal rule of thumb: about 20 training tokens per parameter, one fitted point and not a constant.
- KV-cache bytes: `2 × layers × sequence × kv_heads × head_dim × bytes × batch`. Grouped-query attention lowers `kv_heads`; a sliding-window layer replaces `sequence` by `min(sequence, W)`; a recurrent layer keeps a constant-size state.
- Latent-attention cache bytes: `layers × sequence × (d_c + d_R) × bytes × batch`, with latent width `d_c` and rotary-key width `d_R`.
- Activation tensor bytes: product of dimensions × bytes per element.
- Vector index storage: `documents × dimensions × bytes per value`, before index overhead.
- One optimizer training parameter can require weight + gradient + master weight + first/second moments; account explicitly rather than assuming inference memory.

## Unit discipline

- 1 byte = 8 bits.
- Decimal GB = `10⁹` bytes; binary GiB = `2³⁰` bytes.
- 1 ms = `10⁻³` s.
- Throughput and latency are not reciprocals under batching and concurrency.
- Token counts depend on tokenizer.
- Bits per value of a block-scaled format include its scales.
- Peak FLOP/s depend on the number format and are not delivered throughput.
- Loss and perplexity are not comparable across tokenizers; bits per byte is.

# Appendix B — Glossary

**Acceptance length:** Average number of tokens one verification pass of the target model yields in speculative decoding.

**Access tier:** Level of availability that limits a model or capability to organizations verified through a program.

**Activation:** Intermediate tensor or nonlinear function output in a network.

**Activation checkpointing:** Storing selected activations and recomputing the rest during the backward pass, trading extra arithmetic for lower memory.

**Active parameters:** Parameters used for one token in a sparse model such as a mixture of experts; per-token compute follows them, memory follows the total, and active divided by total is the active fraction.

**Adaptive thinking:** Provider setting in which the model decides for each request whether and how much to reason, steered by an effort level instead of a fixed token budget.

**Advantage:** In policy-gradient reinforcement learning, how much better a response was than a baseline; in GRPO, the reward minus the group mean, divided by the group standard deviation.

**Agent:** A system that uses a model in a loop to observe, choose actions, receive results, and update state.

**Agent Card:** JSON document an A2A agent publishes to describe its identity, capabilities, skills, endpoint, and authentication requirements.

**Agent skill:** Directory containing a `SKILL.md` file and optional scripts and resources that an agent loads progressively when a task matches its description.

**Agent2Agent protocol (A2A):** Open protocol by which independent agents delegate work to each other as peers, using Agent Cards, messages, tasks, and artifacts.

**Agentic AI Foundation (AAIF):** Neutral foundation under the Linux Foundation that hosts open agent standards, including MCP, AGENTS.md, A2A, and Agent Skills.

**Agentic RAG:** Retrieval treated as one tool in an agent loop: the model decides whether to search, writes the query, reads the results, and decides whether to search again.

**AGENTS.md:** Plain Markdown file in a repository that tells coding agents how to build, test, and behave there; one from an untrusted repository is untrusted input.

**AI control:** Safety engineering that assumes a model might try to subvert safeguards and tests whether the safeguards still hold, without needing to know whether the model is aligned.

**Alignment:** Efforts to make behavior accord with intended goals, constraints, and legitimate values.

**Alignment faking:** Complying with a training objective during training to avoid being modified, and behaving differently outside training.

**Annealing:** Final phase of training in which the learning rate falls toward zero; the data seen then has outsized effect.

**Arithmetic intensity:** Arithmetic operations performed per byte moved from memory; work below a device’s ridge point is memory-bound.

**Asymmetric verification:** Property of a question that is hard to answer but cheap to grade.

**Attention:** Content-dependent weighted aggregation of value vectors based on query–key scores.

**Attention sink:** Token, usually an early one, that absorbs surplus attention because softmax weights must sum to one; some architectures add a learned sink explicitly.

**Attribution graph:** Graph from circuit tracing whose nodes are features and whose edges are estimated causal influences for one prompt; a hypothesis to test, not an audit.

**Automatic prefix caching:** Reuse of stored key–value blocks for a request prefix that matches earlier requests, so that part of the prompt is not recomputed.

**Autoregressive:** Factorizes a sequence into predictions conditioned on earlier elements.

**Auxiliary-loss-free balancing:** Mixture-of-experts load balancing that adds a per-expert bias to router scores only when choosing experts, adjusting it by observed load instead of adding a balancing loss.

**Backpropagation:** Reverse-mode differentiation through a computation graph to compute parameter gradients.

**Base model:** A pretrained model before task- or assistant-oriented post-training.

**Batch:** Examples processed together for parallel efficiency.

**Benchmark:** Standardized dataset and protocol used to measure selected behavior.

**Bfloat16:** Sixteen-bit floating-point format with FP32-like exponent range and fewer significand bits.

**Bits per byte:** Tokenizer-independent loss measure: total loss in bits divided by the number of bytes of text.

**Block diffusion:** Generation that proceeds block by block, left to right, denoising within each block, so earlier blocks can be cached.

**Budget forcing:** Controlling reasoning length at inference by ending thinking at the end-of-thinking delimiter, or extending it by suppressing the delimiter and appending “Wait.”

**Byte-pair encoding (BPE):** Subword tokenization based on iteratively merging frequent adjacent units.

**Cache salt:** Per-tenant value mixed into prefix-cache block hashes so that tenants do not share cached blocks.

**Calibration:** Agreement between stated confidence and observed frequency under a protocol.

**Canary:** Fake secret or unique marker planted in a test or dataset so that a leak can be detected mechanically.

**Capability (security):** Metadata attached to a value, such as its sources and allowed readers, that a policy checks when a tool is called; CaMeL uses it to block data flows to unauthorized destinations.

**Capability threshold:** Level of dangerous capability, defined in a frontier safety framework, at which stated safeguards, delays, or pauses apply.

**Cascade:** Multimodal design that chains specialized models and passes text between them, in contrast with a natively multimodal model.

**Causal mask:** Attention mask preventing a position from reading future positions.

**Chain of thought:** Generated intermediate natural-language steps used as scratch space or explanation.

**Checkpoint:** Saved model state, and during training often optimizer/scheduler/data state.

**Chunk:** Segment of a document indexed or placed in context for retrieval.

**Chunked prefill:** Splitting a long prompt into pieces that the scheduler mixes into batches with ongoing decodes, so one long prompt does not stall them.

**Circuit tracing:** Interpretability method that replaces parts of a model with a more interpretable stand-in, such as a cross-layer transcoder, and follows a computation through it.

**Classification:** Map an input to one or more discrete labels.

**Code of practice:** Voluntary document that gives a recognized way to show compliance with duties under the EU AI Act.

**Compaction:** Summarizing an agent’s history when the context window nears its limit and continuing from the summary.

**Compute-optimal:** Allocation of model/data/resources that minimizes a chosen loss under a compute budget.

**Computer-use agent:** Agent that operates software through the interface a person uses, reading screenshots and emitting mouse and keyboard actions.

**Constrained decoding:** Masking tokens that would make the output prefix invalid under a schema or grammar, which guarantees syntax but not meaning.

**Context engineering:** Selecting, ordering, compressing, and securing everything placed in a model’s context at each step.

**Context parallelism:** Divides a long sequence across devices; attention exchanges blocks of keys and values so queries can reach remote ones. Some papers call it sequence parallelism, a name Megatron-LM uses for a narrower technique.

**Context rot:** Degraded use of information as context grows long.

**Context window:** Maximum token sequence accepted by a model or runtime; not a guarantee of perfect use.

**Contextual retrieval:** Prepending a short model-written note that situates each chunk in its document before the chunk is embedded and indexed.

**Continued pretraining:** Additional training on the pretraining-like objective, often using domain data.

**Continuous batching:** Inserting new requests into a running batch as others finish, which improves utilization under variable output lengths.

**Control evaluation:** Test in which a red team builds the strongest attack an untrusted model could mount, reporting both attacks stopped (safety) and legitimate work completed (usefulness).

**Cross-attention:** Queries from one stream attend to keys and values from another.

**Cross-entropy:** Expected negative log-probability assigned by a predicted distribution to target outcomes.

**Cross-layer transcoder:** Model of sparsely active features trained to approximate a network’s MLP blocks, each feature reading one layer and writing to that layer’s MLP output and all later ones; used in circuit tracing.

**Data parallelism:** Replicate model computation over different mini-batches and aggregate gradients.

**Decode phase:** Generation phase that produces tokens one at a time, often limited by memory bandwidth and latency.

**Decoder:** Autoregressive or output-generating component; decoder-only LLMs use causal self-attention.

**Decoding:** Procedure that converts model output distributions into a sequence.

**Decoupled rotary key:** In multi-head latent attention, the small extra key per token that carries rotary position so the cached latent can stay free of position.

**Deep research agent:** Agent that plans a search strategy, runs many searches and page reads, and writes a cited report.

**Deliberative alignment:** Training a model to recall and reason over written safety specifications before answering.

**Delta rule:** Recurrent-state update that first subtracts what the state already predicts for a key, so a new value overwrites the old association instead of piling on.

**Dense model:** Model that activates essentially all relevant weights for each token, in contrast with sparse experts.

**Dense retrieval:** Retrieval with learned embeddings, which captures semantic similarity.

**Diffusion model:** Generative model trained to reverse a progressive noising process or related flow.

**Disaggregated prefill and decode:** Serving that runs prompt processing and token generation on separate groups of GPUs and transfers the KV cache between them.

**Distillation:** Train a student model using outputs or distributions from a teacher.

**Distribution shift:** Difference between training/evaluation data and actual future inputs or relationships.

**DPO:** Direct preference optimization, a reference-anchored objective over preferred/rejected responses.

**Draft head:** Extra output head on a target model that proposes later tokens for the target to verify in speculative decoding; Medusa is an example.

**Dynamic chunking:** Learned grouping of bytes into chunks, trained end to end with the rest of the network, that replaces a fixed tokenizer (H-Net).

**Elicitation:** MCP client feature that lets a server ask the user for more information.

**Embedding:** Learned vector representation of a token, passage, image, or other object.

**Encoder:** Component that forms representations of observed input, often with bidirectional attention.

**Entropy:** Expected surprise of outcomes under a distribution.

**Entropy collapse:** Policy becoming nearly deterministic early in reinforcement learning.

**Entropy patching:** Byte grouping in which a small byte-level model estimates how uncertain the next byte is and starts a new patch at high-uncertainty points (Byte Latent Transformer).

**Epoch:** Approximately one pass through a finite training dataset.

**Evaluation awareness:** Ability of a model to tell evaluation transcripts from real use.

**Evaluation leakage:** Information from test items influences training or model selection.

**Expert:** Subnetwork in a mixture-of-experts layer selected by a router.

**Expert parallelism:** Placing mixture-of-experts experts on different devices and routing tokens to them with all-to-all communication.

**Fine-grained experts:** Mixture-of-experts design that splits large experts into many smaller ones and activates proportionally more of them, keeping compute similar while multiplying possible combinations.

**Fine-tuning:** Additional parameter training to adapt an existing model.

**FlashAttention:** IO-aware exact attention algorithm that reduces expensive memory movement.

**Floating point:** Finite-precision representation with sign, exponent, and significand (the mantissa).

**Flow matching:** Generative training that learns a velocity field carrying noise to data along simple paths; rectified flow uses nearly straight paths.

**Foundation model:** Broadly trained model adapted to many downstream tasks.

**FP8:** Eight-bit floating point. E4M3 (4 exponent, 3 mantissa bits; largest value 448) favors precision; E5M2 (5 exponent, 2 mantissa bits; largest value 57,344) favors range.

**Frontier safety framework:** Developer’s published document stating in advance which dangerous capabilities it will test for, what counts as a capability threshold, and which safeguards, delays, or pauses follow.

**Full-duplex model:** Speech model that listens and speaks at the same time, treating the user’s speech and its own as parallel streams.

**Fully open model:** Release that publishes the data, code, checkpoints, and other pieces behind training, not only weights, so others can audit and reproduce it.

**Gated delta rule:** Delta rule with a learned decay gate: values near 0 clear memory quickly, values near 1 leave only targeted overwrites.

**General-purpose AI model:** Under the EU AI Act, a model with broad capability that can be built into many downstream systems; its provider has documentation, copyright-policy, and training-summary duties.

**General-purpose model with systemic risk:** Under the EU AI Act, a general-purpose model presumed to have high-impact capabilities, notably when its training exceeds 10²⁵ floating-point operations; its provider has further duties.

**Goodput:** Rate of requests that meet stated quality and latency targets, such as both a first-token and a per-token target.

**Gradient:** Vector of local loss sensitivities with respect to parameters.

**Gradient accumulation:** Sum gradients over micro-batches before one optimizer update.

**Gradient clipping:** Rescaling the whole gradient when its global norm exceeds a threshold, guarding against rare huge steps.

**Grounding:** Connecting an output to provided evidence, perception, tools, or the external world.

**Grouped-query attention (GQA):** Attention in which each group of query heads shares one key head and one value head, shrinking the KV cache by the group size; one group is multi-query attention.

**GRPO:** Group relative policy optimization: a PPO-style method with no value model, whose baseline is the average reward of several responses sampled for the same prompt.

**Hallucination:** Broad term for unsupported generated content; diagnose the specific failure.

**Hard negative:** Wrong candidate that scores close to the positive in contrastive training, such as a passage that shares many words with the question but does not answer it.

**HBM:** High-bandwidth memory: stacked DRAM placed on the same package as an accelerator and read through a very wide interface.

**Head:** One parallel attention subspace within multi-head attention.

**Hidden state:** Internal representation at a layer and position.

**Hybrid model:** Model that mixes mostly recurrent or linear-attention layers with a few full-attention layers, so most layers keep a constant-size state.

**Importance ratio:** New-policy probability of a token divided by the old policy’s; PPO-style objectives clip it to 1 ± ε so that one update cannot move the policy too far.

**Inference:** Running a trained model without ordinary parameter updates to produce outputs.

**InfoNCE:** Contrastive loss: a softmax cross-entropy over one positive and several negatives, with similarity scores divided by a temperature.

**Inoculation prompting:** Framing a training behavior, such as reward hacking, as acceptable so that it generalizes less into broader misalignment.

**Instruction tuning:** Fine-tuning on examples that demonstrate following tasks or conversational roles.

**Jailbreak:** Input designed to bypass intended model behavior restrictions.

**Key:** Attention vector matched against a query.

**Knowledge cutoff:** Date or range after which a model’s parametric knowledge is not expected to be reliable.

**KV cache:** Stored attention keys and values for previous tokens during autoregressive generation.

**Late interaction:** Retrieval scoring that keeps one vector per token and, for each query token, takes its best match among the document’s tokens (MaxSim), then sums over query tokens; ColBERT is an example.

**Latency:** Time for a request or stage; report distribution and workload.

**Latent diffusion:** Diffusion run in the compressed latent space of an autoencoder instead of on raw pixels, with a decoder turning the final latent back into an image.

**Layer normalization:** Normalizes features within an example/token, then applies learned scale and shift.

**Learning rate:** Optimizer step-size factor.

**Lethal trifecta:** Three properties that together make an agent exploitable: access to private data, exposure to untrusted content, and the ability to communicate externally.

**Lightning indexer:** Cheap scoring module, with few heads and low precision, that picks the top-k earlier tokens each query attends to in DeepSeek Sparse Attention.

**Linear attention:** Attention without softmax, rearranged so that a fixed-size state is updated per token, giving constant cost and memory per generated token at the price of limited recall.

**LLM-as-judge:** Use of a language model to score or compare outputs under a rubric.

**Local–global interleaving:** Layer pattern that mixes sliding-window layers with occasional full-attention layers.

**Logit:** Unnormalized score before softmax or another link function.

**LoRA:** Low-rank adaptation that trains a factorized update to frozen weight matrices.

**Loss:** Scalar training objective to minimize.

**Loss scaling:** Multiplying the loss by a constant before backpropagation so 16-bit gradients stay in range, then undoing it before the update.

**Mask:** Structure that disallows selected attention connections or loss positions.

**Masked diffusion language model:** Model trained to recover randomly masked tokens at a random mask ratio, with the loss weighted by the inverse of that ratio; it generates by gradually unmasking a fully masked sequence.

**Matryoshka representation learning:** Training an embedding so that each of several prefix lengths works as an embedding on its own, which lets you truncate vectors.

**Maximal update parametrization (μP):** Parametrization in which good learning rates and other hyperparameters stay stable as width grows, so values tuned on a small model transfer to a larger one.

**Mechanistic interpretability:** Study of internal features and circuits that causally implement behavior.

**Memory bandwidth:** Rate at which data move to/from memory; often an inference bottleneck.

**Memory file:** Notes an agent writes outside its context window and reads back after a reset or compaction.

**Microscaling (MX):** Open Compute Project block-scaled formats (MXFP8, MXFP6, MXFP4, MXINT8) in which a block of 32 narrow elements shares one 8-bit power-of-two scale; version 1.1 of the specification renames it MicroXcaling.

**Mid-training:** Late pretraining phase, often combined with annealing, that switches to a smaller, higher-quality, more targeted data mixture.

**Mini-batch:** Subset of examples used to estimate a gradient.

**Mixture of experts (MoE):** Architecture routing each token through a subset of expert subnetworks.

**Model card:** Documentation of intended use, evaluation, limitations, and relevant details of a model.

**Model collapse:** Degradation of models trained recursively on model-generated data, as the tails of the original data distribution disappear.

**Model Context Protocol (MCP):** Open standard for connecting AI applications to tools and data: a host application runs one client per connection to a server that offers tools, resources, and prompts.

**Model merging:** Combining the weights of models that share an ancestor into one model without further gradient training.

**Monitorability:** Ability of a monitor to predict properties of an agent’s behavior; it depends on both the monitor and the monitored agent.

**Monitorability tax:** Performance given up to keep reasoning monitorable, such as by not applying strong optimization pressure to the chain of thought.

**Multi-head latent attention (MLA):** Attention that caches one small latent vector per token, plus a small decoupled rotary key, and rebuilds per-head keys and values from it.

**Multi-query attention (MQA):** Attention in which all query heads share a single key head and a single value head.

**Multi-token prediction (MTP):** Training objective, also usable for drafting in speculative decoding, in which extra heads or modules predict tokens beyond the next one.

**Multimodal:** Processes or generates more than one modality, such as text, image, audio, or video.

**Muon:** Optimizer that replaces the momentum matrix of each hidden-layer weight matrix with the nearest semi-orthogonal matrix, approximated by a few Newton–Schulz iterations.

**MXFP4:** Microscaling format with 4-bit E2M1 elements in blocks of 32 that share one power-of-two scale, 4.25 bits per value.

**N-gram:** Sequence of n units; count-based language models condition on the previous n−1 units.

**Natively multimodal:** Trained so that one network reads and writes several modalities directly, instead of passing text between specialized models.

**Next-token prediction:** Objective of predicting the following token from a prefix.

**NoPE:** No positional encoding; a decoder relies on the causal mask to carry order.

**Normalization:** Controls scale or distribution of activations for optimization.

**Nucleus sampling:** Samples from the smallest token set whose cumulative probability reaches top-p.

**NVFP4:** NVIDIA 4-bit format with E2M1 elements in blocks of 16, an FP8 (E4M3) scale per block, and one FP32 scale per tensor; 4.5 bits per value before the tensor scale.

**Omni model:** Model trained end to end to accept and produce several modalities; an “omni” product may instead orchestrate several specialized models.

**One-hot:** Vector with one 1 and all other components zero, representing a category.

**Open-weight model:** Model whose trained parameters are published for download; the training data, code, recipe, and license terms may still be withheld or restricted.

**Optimizer:** Algorithm that converts gradients and state into parameter updates.

**Over-training:** Training a model on more tokens than is compute-optimal for its size, to lower the cost of serving it.

**Overfitting:** Fit to training peculiarities that does not transfer adequately.

**Overthinking:** A reasoning model spending many tokens on trivial questions for little benefit.

**p-RoPE:** Rotary embeddings applied to only a fraction p of the frequency pairs, dropping the lowest frequencies; p = 0 is NoPE.

**Paged KV cache:** KV cache divided into blocks mapped through a table, like virtual memory, so variable-length sequences do not fragment memory.

**Parameter:** Learned numerical value in a model.

**Pass@k:** Fraction of problems solved by at least one of k independent samples.

**Pass^k:** Probability that all k independent trials succeed; it measures reliability, whereas pass@k asks whether any trial succeeds.

**Perplexity:** Exponential of average token cross-entropy in nats.

**Pipeline parallelism:** Splits layers into stages and streams micro-batches through them like an assembly line; idle stage time is a pipeline bubble.

**Policy:** In reinforcement learning, a distribution over actions given state; an LLM can be treated as a token policy.

**Polysemantic:** Responding to several unrelated concepts, as single neurons often do under superposition.

**Position interpolation:** Context-extension method that divides position indices by the extension factor so rotary angles stay within the range seen in training.

**Post-training:** Training after base pretraining to shape tasks, preferences, reasoning, safety, or tools.

**PPO:** Proximal policy optimization: a reinforcement-learning algorithm that limits each update with a clipped ratio of new to old token probabilities and usually estimates its baseline with a learned value model.

**Precision:** Numerical format or number of represented digits/bits; also a classification metric.

**Prefill:** Prompt-processing phase that computes hidden states and fills KV cache.

**Progressive disclosure:** Loading only an agent skill’s name and description at startup and its full instructions when a task matches.

**Prompt injection:** Attempt to make untrusted content override intended instructions or cause unsafe actions.

**QK-Clip:** Rescaling a head’s query and key projection weights after an update when its largest attention logit exceeds a threshold.

**QK-normalization:** Normalizing queries and keys before their dot product to prevent huge attention logits.

**Quantization:** Represent weights/activations/caches with fewer or discrete numeric levels.

**Query:** Attention vector expressing what information a destination position seeks.

**RadixAttention:** KV-cache reuse scheme, introduced with the SGLang runtime, that stores the caches of all requests in a radix tree with least-recently-used eviction, making the longest cached prefix cheap to find.

**RAG:** Retrieval-augmented generation, which supplies external evidence to a generator.

**Random Hadamard transform:** Fixed orthogonal mixing with random signs, applied to tensors to spread large outliers over many values.

**ReAct:** Agent pattern that interleaves reasoning-like text and actions.

**Reasoning effort:** Provider setting that steers how much a model thinks before answering; a dial for quality, latency, and cost, not a guarantee.

**Reasoning model:** Loose term for a model post-trained, usually with reinforcement learning, to deliberate at length before answering, or for a product that exposes such a mode; it is not a distinct architecture.

**Red teaming:** Adversarial search for failures, harms, or policy circumvention.

**Reinforcement learning:** Learning a policy from consequences or rewards over actions/trajectories.

**Residual connection:** Adds a block’s input to its transformed output.

**Reward hacking:** Raising the measured reward without achieving the intended goal.

**Reward model:** Learned function scoring outputs or trajectories, often from preferences.

**RL environment:** Tasks or a task generator, together with the interface the model acts through, the reward or verifier, and the sandbox that runs them.

**RLHF:** Reinforcement learning from human feedback.

**RLVR:** Reinforcement learning with verifiable rewards: a program checks each sampled response, and the reward is a fixed value if it passes and zero otherwise.

**RMSNorm:** Normalization based on root mean square without mean subtraction.

**Roofline model:** Plot of achievable speed against arithmetic intensity; the ridge point where the compute and memory limits meet separates memory-bound from compute-bound work.

**RoPE:** Rotary position embeddings that rotate query/key features by position.

**Router:** Model or layer that selects models, tools, or experts.

**Rule of Two:** Meta’s proposal that an agent in one session should satisfy at most two of three properties: processing untrustworthy inputs, accessing sensitive systems or private data, and changing state or communicating externally.

**Sampling:** Random selection according to a distribution.

**Sandbagging:** Deliberately underperforming on an evaluation.

**Scale factor:** Number a tensor is multiplied by before conversion to a narrow format and divided by afterward.

**Scale-out network:** Link between nodes across a data center through adapters and switches; it reaches far more devices but gives each far less bandwidth.

**Scale-up network:** Link among accelerators within one node or rack, with the highest bandwidth and lowest latency, which can let one device access another’s memory directly.

**Scaling law:** Empirical relationship between performance/loss and resources over a measured regime.

**Scheming:** Covertly pursuing misaligned goals while hiding one’s capabilities or objectives.

**Self-attention:** Attention where queries, keys, and values derive from the same sequence.

**Self-supervised:** Creates learning targets from the data itself.

**SFT:** Supervised fine-tuning.

**Shampoo:** Matrix-preconditioned optimizer that multiplies the gradient on both sides by matrices built from gradient statistics.

**Shared expert:** Mixture-of-experts expert that processes every token, so common knowledge need not be duplicated across routed experts.

**Sliding-window attention:** Each position attends only to the previous W positions, so the cache can be a fixed-size rolling buffer.

**SOAP:** Optimizer that runs Adam in the eigenbasis of Shampoo’s preconditioner.

**Softmax:** Converts logits to a normalized categorical distribution.

**Sparse attention:** Attention in which each query reads a learned or computed subset of earlier tokens instead of all of them.

**Sparse autoencoder:** Learns a reconstruction using a large set of sparsely active features.

**Sparse retrieval:** Lexical matching such as BM25, strong for names, identifiers, and rare keywords.

**Speculative decoding:** Drafts with a cheaper mechanism and verifies with a target model to accelerate generation.

**State-space model:** Sequence model maintaining a state updated across positions, often with linear-scaling goals.

**Stochastic rounding:** Rounding a value up or down at random, with probabilities set by its closeness to each neighbor, so rounding errors average out instead of biasing the result.

**Structured outputs:** Feature in which the application supplies a schema and the serving system constrains decoding so that the output conforms to it.

**Subnormal number:** Floating-point value stored without the implicit leading 1, so values fade gradually toward zero.

**Superposition:** Representing more features than there are neurons by assigning them to overlapping directions.

**Superword token:** Token that spans a whitespace boundary, as produced by SuperBPE.

**SwiGLU:** Gated feed-forward form in which a SiLU-activated projection multiplies another projection elementwise before the output projection.

**System card:** Documentation of a deployed model/system, evaluations, safeguards, and limitations.

**Taint rule:** Runtime rule that refuses side-effecting tools once untrusted content has entered the conversation.

**Task vector:** Difference between fine-tuned and pretrained weights; adding it can add a skill, and negating it can reduce a behavior.

**Temperature:** Logit scaling that controls sharpness of a sampling distribution.

**Tensor:** Multidimensional numeric array.

**Tensor parallelism:** Splits individual tensor operations across devices.

**Test-time compute:** Computation spent during inference through deliberation, search, samples, or tools; sequential scaling lengthens one chain, and parallel scaling samples many attempts and selects among them.

**Time horizon:** Human-expert duration of the tasks an agent completes with 50 percent probability, read from a logistic fit of success against the logarithm of human time.

**Time per output token (TPOT):** Average delay between generated tokens during decode; also called inter-token latency.

**Time to first token (TTFT):** Delay from a request to its first generated token; it depends on prefill and queueing.

**Token:** Integer-indexed unit produced by a tokenizer; may be a byte, character, subword, code, or special marker.

**Token passthrough:** Passing a client’s token through to a downstream API without checking that it was issued for that server; documented as an MCP anti-pattern.

**Tokenizer:** Encoder/decoder between raw sequences and token IDs.

**Tool call:** Structured model proposal to invoke an external function.

**Tool search:** Loading tool definitions on demand by searching a catalog instead of placing every definition in context.

**Tool-result clearing:** Replacing old tool outputs in an agent’s history with short placeholders.

**Training:** Process that updates model parameters to reduce objectives.

**Transformer:** Architecture built from attention, feed-forward transformations, residuals, normalization, and position handling.

**UE8M0:** Unsigned 8-bit scale type with eight exponent bits and no mantissa, so it stores only powers of two; the shared scale in microscaling formats.

**Under-trained token:** Vocabulary entry that almost never appears in training data, so its embedding is barely updated and it can provoke odd behavior.

**Value:** Attention content vector combined according to query–key weights.

**Vector database:** System indexing vectors for nearest-neighbor retrieval, usually with metadata filters.

**Verifier:** Mechanism that checks a candidate result or process.

**Vision-language-action (VLA) model:** Model that takes camera images and a language instruction and outputs robot actions.

**Warmup–stable–decay (WSD):** Learning-rate schedule that rises from near zero, holds constant for most of training, then decays over a short final phase.

**Weight decay:** Regularization that shrinks parameters, decoupled from adaptive updates in AdamW.

**World action model:** World model coupled directly to action selection.

**World model:** Model that predicts how an environment will change, including how the agent’s own actions change it.

**YaRN:** Context-extension recipe that leaves fast-rotating rotary pairs unchanged, interpolates slow ones, blends those between, and rescales attention logits slightly.

# Appendix C — Myth Versus Precise Statement

| Myth | More precise statement |
|---|---|
| “It is just autocomplete.” | Next-token prediction is the objective; the learned function can implement rich representations and procedures, while still lacking guarantees. |
| “It searches its training database.” | Parametric recall is distributed generation, not ordinary record lookup; applications may separately use retrieval. |
| “Attention shows what the model thought.” | Attention is one causal route; weights alone are not guaranteed to be faithful explanations. |
| “A bigger context is memory.” | It is temporary input capacity with imperfect utilization; durable memory requires an external or learned update mechanism. |
| “Temperature makes it creative.” | Temperature changes distribution sharpness and diversity, not knowledge or value. |
| “Lower loss means safer.” | Predictive loss and safety objectives differ; system controls matter. |
| “Open source means all details are open.” | Weights, code, data, recipe, license, and governance are separate dimensions. |
| “RAG prevents hallucination.” | RAG supplies evidence but can fail at retrieval, interpretation, citation, or abstention. |
| “JSON output is safe.” | Grammar validity does not imply semantic validity or authorization. |
| “The model refused, so the system is secure.” | Refusal is probabilistic behavior; enforce security outside the model. |
| “The chain of thought reveals the real reasoning.” | Generated rationale may help work but is not guaranteed faithful to internal causation. In measured tests, hints that changed an answer were often not mentioned, and a provider’s reasoning summary is itself generated text. |
| “Agents are models with goals.” | Agency is produced by the model–tool–state loop, permissions, and stop conditions. |
| “Fine-tuning adds facts cleanly.” | It changes distributed parameters and may have side effects; retrieval is often better for mutable facts. |
| “Quantization only makes a model smaller.” | It changes numerical computation and can degrade selected tasks or hardware behavior. |
| “A benchmark score is intelligence.” | It is evidence about a specified sample and protocol, vulnerable to contamination and construct gaps. |
| “Human review solves risk.” | Review works only with authority, time, competence, information, and manageable workload. |
| “Reasoning models just think longer, and more thinking is always better.” | Reasoning behavior comes from post-training, often reinforcement learning against checkers, plus an inference-time budget. Studies report overthinking, and effort settings are soft guidance: evaluate them on your own tasks for quality, latency, and cost. |
| “Reinforcement learning teaches a model to reason from scratch.” | Reported evidence disagrees. RL on verifiable rewards raises single-sample accuracy and changes how reasoning is organized, while studies differ on whether it adds abilities the base model lacks. The base model matters, and the question is unsettled. |
| “A verifiable reward means the model learns what you meant.” | It learns what the checker rewards. Incomplete tests, answer matchers, and model judges can be exploited, so audit the verifier, hold out evaluations that were never optimized against, and inspect the highest-reward samples. |
| “Training against a reasoning monitor makes the reasoning safe.” | Light optimization helped, but heavy optimization against a chain-of-thought monitor produced hidden intent while the hacking continued. Treat monitoring as one layer beside action monitoring and containment. |
| “The context window is the usable context.” | A window is the upper bound the interface accepts. Measured use degrades with length and varies by task, so test the real workload at the real length. |
| “Long context makes retrieval obsolete.” | Studies point both ways. Long context can win on small, stable corpora at higher cost; retrieval suits large, changing, or permissioned corpora and supplies citations. Decide by testing. |
| “A mixture-of-experts model costs only its active parameters.” | Per-token arithmetic follows the active parameters, but memory follows the total, a batch touches many experts, and tokens must travel to them. |
| “FlashAttention approximates attention or makes it linear.” | It computes exact attention and avoids storing the full score matrix, so extra memory grows linearly while the arithmetic stays quadratic. |
| “Four-bit means four bits per value.” | Block-scaled formats add scales (4.25 bits per value for MXFP4, 4.5 for NVFP4), recipes keep some tensors in higher precision, and peak throughput is not delivered speed. |
| “Speculative decoding trades quality for speed.” | With exact verification the output distribution equals the target model’s. The speedup depends on acceptance and shrinks at large batch sizes; variants that relax the rule can change outputs. |
| “Chinchilla’s 20 tokens per parameter is the best recipe.” | It is one fitted point for minimizing training compute. Models that will be served heavily are trained on far more tokens to cut inference cost. |
| “Training on synthetic data always collapses a model.” | Collapse is reported when generated data replaces real data; in experiments, accumulating it alongside real data avoided collapse. Keep real data in the mix and track diversity. |
| “A standard protocol such as MCP makes tools safe.” | A protocol standardizes the connection. The MCP specification says it cannot enforce its security principles at the protocol level, so consent flows, authorization, access controls, and treating tool descriptions as untrusted remain the implementor’s job. |
| “A model that resists injection in tests is a security boundary.” | Training helps, but defenses reported to block nearly all attacks have fallen to adaptive attackers. Limit what an agent can do: avoid combining private data, untrusted content, and external communication, and enforce policy in code. |
| “An agent that usually succeeds is reliable.” | Reliability asks that every repeated trial succeed (pass^k), and long workflows compound errors: if each of 20 independent steps succeeds with probability 0.98, the whole path succeeds about 67 percent of the time. |
| “Instruction files and skills are safe configuration.” | They are text an agent will follow. Files and skills from untrusted sources are untrusted input and part of your software supply chain. |
| “An omni model is one model.” | “Omni” can mean one network trained end to end, or a product that orchestrates several specialist models behind one interface. Ask which component handles each modality. |
| “A passing evaluation shows how the model will behave in deployment.” | Models can sometimes tell evaluations from real use, and graders can be exploited. Make tests realistic, inspect trajectories, and monitor real deployments. |
| “Once published, a benchmark stays valid.” | Benchmarks age through leakage and test errors; one developer stopped reporting a widely used software-engineering benchmark after an audit. Refresh, hold out, or write original items. |

# Appendix D — Operational Checklists

## Before choosing a model

- Define task, population, error costs, and authority.
- Build a private representative evaluation.
- Establish a simple non-LLM or small-model baseline.
- Measure quality, latency, cost, privacy, and operations.
- Verify license, data terms, region, retention, and deprecation policy.
- Test at realistic context, concurrency, language, and tool use.
- Plan fallback and rollback.
- Pin the exact model version, and re-run your evaluation when it changes or is retired.
- Compare total and active parameters, advertised and measured context, and output limits.
- If the model reasons, test its effort setting on your tasks and budget for reasoning tokens and latency.
- Check which tools run in the provider’s sandbox and which run on your infrastructure.

## Before adding retrieval

- Confirm knowledge freshness or attribution is the actual bottleneck.
- Preserve source, section/page, timestamp, owner, and permissions.
- Evaluate parser quality and retrieval recall separately.
- Include unanswerable and conflicting-source cases.
- Apply authorization filters inside the retrieval query, before context assembly, and recheck each candidate’s current permissions.
- Defend against indirect prompt injection.
- Validate citations and deletion propagation.
- Compare against placing a small, stable, readable corpus in context; decide by testing on your own questions.
- Build a lexical baseline before adding dense retrieval.
- When retrieval is iterative or agentic, evaluate the trajectory (queries issued, sources opened) as well as the answer.

## Before granting a tool

- Is the tool necessary?
- Can it be read-only or preview-only?
- Does it use authenticated scope instead of model-supplied identity?
- Are arguments typed, bounded, and semantically validated?
- Are network/filesystem/credentials minimized?
- Are timeout, rate, cost, and step budgets enforced?
- Are writes idempotent and confirmed?
- Is every action auditable and reversible?
- Can malicious tool output influence later actions?
- Is each tool server, skill, or instruction file from a publisher you trust and pinned to a reviewed version, with its descriptions treated as untrusted input?
- Does each token work only for the server it was issued for, never passed through to another service?
- Does any session combine private data, untrusted content, and external communication? Remove one of the three or require human approval.
- Does generated code run only in an isolated sandbox without credentials?
- Is the tool catalog small and clearly named, and is tool selection evaluated?

## Before fine-tuning

- Freeze a held-out evaluation and base-model baseline.
- Audit provenance, consent/license, duplicates, quality, and privacy.
- Match the exact tokenizer and chat template.
- Specify loss masking and sequence truncation.
- Track train/validation curves and checkpoints.
- Test general capability retention, safety, and memorization.
- Document adapter/base compatibility and deployment precision.
- If you merge models, test every source capability on the merged model and check each source license.

## Before using reinforcement learning with verifiable rewards

- Is there a checker you trust? Audit it for false negatives, loopholes, and incomplete tests.
- Keep tests, scorers, and answer keys out of the agent’s reach, and verify outcomes from environment state, not from the model’s claims.
- Run generated code only in an isolated sandbox with no credentials.
- Compare against prompting and supervised fine-tuning baselines.
- Hold out evaluations that are never optimized against, and stop when held-out results diverge from the training reward.
- Track response length, entropy, and the share of prompts whose sampled answers all score the same, since those prompts give no learning signal.
- Hand-check the highest-reward samples.
- Report pass@1 and pass@k at large k, on more than one base-model family, with reward ablations.
- Do not apply strong optimization pressure to the chain of thought itself; keep monitors as a separate signal.
- Decontaminate training prompts against your evaluations.

## Before changing precision or the serving stack

- Reproduce vendor quality comparisons on your own tasks, long contexts, and languages.
- Confirm the hardware supports the format natively; weight-only and weight-and-activation formats differ.
- Size the KV cache for your target context and concurrency, and test long-context retrieval after quantizing it.
- Compare numerical outputs and task quality after any kernel or compiler change.
- For speculative decoding, confirm verification is exact and measure acceptance length at production batch size.
- Scope prefix and response caches by tenant, model version, and policy, and treat response timing as an output that can leak.
- Report hardware, batch size, prompt and output lengths, and precision with any speed claim, and track goodput, not only tokens per second.

## Before trusting an evaluation result

- Name the decision the result will inform.
- Record benchmark version and date, harness, tool and token budgets, and report spread over repeated runs.
- Check contamination defenses: refreshed, held-out, or original items.
- Audit the answer key and tests; a miss may be a bad item.
- Validate any model judge against human labels, and check position, verbosity, and self-preference bias.
- For agents, report reliability (pass^k) as well as single-run success, and read trajectories for grader exploits.
- Make environments realistic, because models may recognize tests, and keep some items private.

## Before launch

- Complete threat model, privacy and impact review.
- Define model/system card and user disclosure.
- Red-team direct and indirect injection.
- Load-test realistic traffic and worst-case lengths.
- Verify logs do not collect unnecessary sensitive content.
- Set alerts for errors, cost, latency, refusals, and policy events.
- Stage rollout with kill switch and rollback.
- Assign incident owners and user recourse.
- Run code-executing agents, including during evaluation, in strong isolation.
- Identify your legal role (provider or deployer) and the obligations that apply in each jurisdiction on the launch date.
- Do not base safety or audit claims on reasoning text alone.
- Decide what evidence triggers suspension.

# Appendix E — A One-Year Curriculum

## Months 1–2: computation and mathematics

Learn Python basics, data structures, command line, version control, vectors/matrices, probability, derivatives, and optimization. Learn how numbers are stored (floating-point formats such as FP32, BF16, and FP8) and why moving data, not arithmetic, often limits speed. Implement sampling (Lab 0), scalar autodiff (Lab 2), and linear regression fitted by gradient descent.

**Exit test:** derive softmax cross-entropy for a tiny example, gradient-check a two-layer network, and compute how long one pass over a model’s weights takes at a stated memory bandwidth.

## Months 3–4: machine learning and neural networks

Study train/validation/test design, generalization, MLPs, normalization, CNN/RNN history, embeddings and how they are trained (contrastive loss with hard negatives), optimizers and learning-rate schedules (AdamW, warmup–stable–decay), and sequence objectives. Reproduce a small classification experiment with proper uncertainty.

**Exit test:** diagnose underfitting, overfitting, leakage, and shift in a controlled dataset.

## Months 5–6: transformers

Implement tokenization, one attention head, and a tiny decoder (Labs 1, 3, and 4). Read the Transformer paper [R07] and annotated implementations. Measure ablations. Then study why recent architectures change attention (grouped-query and latent attention, sliding windows, rotary position scaling, mixture of experts, linear-attention hybrids) and size their caches (Lab 9).

**Exit test:** trace every tensor shape through a transformer block, and estimate total parameters, active parameters, and KV-cache memory from a published architecture table, counting each layer type separately.

## Months 7–8: post-training and applications

Study SFT, LoRA, preference optimization, reinforcement learning with verifiable rewards (Lab 10), reasoning models and test-time compute, decoding and structured output, embeddings, RAG with access control, context engineering, tools, and protocols such as MCP. Adapt a small model, build a retrieval baseline, and write a safe tool loop (Labs 5, 6, and 7).

**Exit test:** explain when to prompt, retrieve, fine-tune, distill, or train from scratch using measured evidence; for a task with an automatic checker, say when reinforcement learning fits and how the checker could be gamed.

## Months 9–10: systems and evaluation

Learn GPU memory hierarchy, arithmetic intensity, batching, quantization and low-precision formats, serving (KV-cache management, prefix caching, speculative decoding), distributed training concepts, evaluation design (contamination, judge reliability, agent reliability), and statistics. Build an evaluation harness (Lab 8) and benchmark one model in two precisions.

**Exit test:** report quality–latency–cost trade-offs with a reproducible evaluation card that names hardware, batch size, and precision.

## Months 11–12: safety, interpretability, and capstone

Threat-model an agent that uses tools and protocols, conduct red-team tests, study governance (frontier safety frameworks, AI control, the laws that apply where you work) and interpretability, run an agent loop under attack (Lab 11), and complete the capstone service.

**Exit test:** demonstrate bounded authority, an injection test that fails when the defense is removed, incident rollback, and an honest system card; present known failures before successes.

## Ongoing

Read one primary paper deeply each week, reproduce one result monthly, contribute documentation or tests to an open project, check dated claims (model names, versions, legal deadlines) against official sources before relying on them, and revise your beliefs publicly when evidence changes.

# References and Primary Reading

The bibliography prioritizes primary papers and living official frameworks. A citation supports a specific idea, not every claim in the surrounding chapter. Preprints may change; consult later peer-reviewed versions where available.

## Origins, neural networks, and language

**[R01]** Alan M. Turing. “Computing Machinery and Intelligence.” *Mind* 59(236), 1950. [doi:10.1093/mind/LIX.236.433](https://doi.org/10.1093/mind/LIX.236.433).

**[R02]** David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams. “Learning Representations by Back-Propagating Errors.” *Nature* 323, 1986. [doi:10.1038/323533a0](https://doi.org/10.1038/323533a0).

**[R03]** Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Jauvin. “A Neural Probabilistic Language Model.” *JMLR* 3, 2003. [jmlr.org/papers/v3/bengio03a.html](https://www.jmlr.org/papers/v3/bengio03a.html).

**[R04]** Tomas Mikolov et al. “Efficient Estimation of Word Representations in Vector Space.” 2013. [arXiv:1301.3781](https://arxiv.org/abs/1301.3781).

**[R05]** Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. “Sequence to Sequence Learning with Neural Networks.” 2014. [arXiv:1409.3215](https://arxiv.org/abs/1409.3215).

**[R06]** Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. “Neural Machine Translation by Jointly Learning to Align and Translate.” 2014. [arXiv:1409.0473](https://arxiv.org/abs/1409.0473).

## Transformers, representation, and scale

**[R07]** Ashish Vaswani et al. “Attention Is All You Need.” 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).

**[R08]** Jacob Devlin et al. “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.” 2018. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805).

**[R09]** Tom B. Brown et al. “Language Models are Few-Shot Learners.” 2020. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165).

**[R10]** Colin Raffel et al. “Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.” *JMLR* 21, 2020. [jmlr.org/papers/v21/20-074.html](https://www.jmlr.org/papers/v21/20-074.html).

**[R11]** Jared Kaplan et al. “Scaling Laws for Neural Language Models.” 2020. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361).

**[R12]** Jordan Hoffmann et al. “Training Compute-Optimal Large Language Models.” 2022. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556).

## Instruction and preference foundations

**[R13]** Long Ouyang et al. “Training Language Models to Follow Instructions with Human Feedback.” 2022. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155).

**[R14]** Rafael Rafailov et al. “Direct Preference Optimization: Your Language Model Is Secretly a Reward Model.” 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290).

**[R15]** Yuntao Bai et al. “Constitutional AI: Harmlessness from AI Feedback.” 2022. [arXiv:2212.08073](https://arxiv.org/abs/2212.08073).

## Optimization, tokenization, and architecture

**[R16]** Ilya Loshchilov and Frank Hutter. “Decoupled Weight Decay Regularization.” 2017. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101).

**[R17]** Rico Sennrich, Barry Haddow, and Alexandra Birch. “Neural Machine Translation of Rare Words with Subword Units.” 2015. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909).

**[R18]** Taku Kudo and John Richardson. “SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing.” 2018. [arXiv:1808.06226](https://arxiv.org/abs/1808.06226).

**[R19]** Tri Dao et al. “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.” 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135).

**[R20]** Jianlin Su et al. “RoFormer: Enhanced Transformer with Rotary Position Embedding.” 2021. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864).

**[R21]** Ofir Press, Noah A. Smith, and Mike Lewis. “Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation.” 2021. [arXiv:2108.12409](https://arxiv.org/abs/2108.12409).

**[R22]** Joshua Ainslie et al. “GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.” 2023. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245).

**[R23]** Noam Shazeer et al. “Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.” 2017. [arXiv:1701.06538](https://arxiv.org/abs/1701.06538).

**[R24]** William Fedus, Barret Zoph, and Noam Shazeer. “Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.” 2021. [arXiv:2101.03961](https://arxiv.org/abs/2101.03961).

**[R25]** Albert Gu and Tri Dao. “Mamba: Linear-Time Sequence Modeling with Selective State Spaces.” 2023. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752).

**[R26]** Samyam Rajbhandari et al. “ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.” 2019. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054).

## Preference learning, reasoning, and generation

**[R27]** John Schulman et al. “Proximal Policy Optimization Algorithms.” 2017. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347).

**[R28]** Jason Wei et al. “Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.” 2022. [arXiv:2201.11903](https://arxiv.org/abs/2201.11903).

**[R29]** Xuezhi Wang et al. “Self-Consistency Improves Chain of Thought Reasoning in Language Models.” 2022. [arXiv:2203.11171](https://arxiv.org/abs/2203.11171).

**[R30]** Hunter Lightman et al. “Let’s Verify Step by Step.” 2023. [arXiv:2305.20050](https://arxiv.org/abs/2305.20050).

**[R31]** Charlie Snell et al. “Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Model Parameters.” 2024. [arXiv:2408.03314](https://arxiv.org/abs/2408.03314).

**[R32]** DeepSeek-AI et al. “DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.” 2025. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948).

**[R33]** Ari Holtzman et al. “The Curious Case of Neural Text Degeneration.” 2019. [arXiv:1904.09751](https://arxiv.org/abs/1904.09751).

**[R34]** Yaniv Leviathan, Matan Kalman, and Yossi Matias. “Fast Inference from Transformers via Speculative Decoding.” 2022. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192).

**[R35]** Edward J. Hu et al. “LoRA: Low-Rank Adaptation of Large Language Models.” 2021. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).

**[R36]** Tim Dettmers et al. “QLoRA: Efficient Finetuning of Quantized LLMs.” 2023. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314).

## Retrieval, agents, and multimodality

**[R37]** Patrick Lewis et al. “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.” 2020. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401).

**[R38]** Vladimir Karpukhin et al. “Dense Passage Retrieval for Open-Domain Question Answering.” 2020. [arXiv:2004.04906](https://arxiv.org/abs/2004.04906).

**[R39]** Shunyu Yao et al. “ReAct: Synergizing Reasoning and Acting in Language Models.” 2022. [arXiv:2210.03629](https://arxiv.org/abs/2210.03629).

**[R40]** Alexey Dosovitskiy et al. “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.” 2020. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929).

**[R41]** Alec Radford et al. “Learning Transferable Visual Models From Natural Language Supervision.” 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020).

**[R42]** Jonathan Ho, Ajay Jain, and Pieter Abbeel. “Denoising Diffusion Probabilistic Models.” 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239).

## Inference systems and efficiency

**[R43]** Elias Frantar et al. “GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.” 2022. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323).

**[R44]** Guangxuan Xiao et al. “SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.” 2022. [arXiv:2211.10438](https://arxiv.org/abs/2211.10438).

**[R45]** Ji Lin et al. “AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.” 2023. [arXiv:2306.00978](https://arxiv.org/abs/2306.00978).

**[R46]** Woosuk Kwon et al. “Efficient Memory Management for Large Language Model Serving with PagedAttention.” 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180).

## Interpretability, documentation, evaluation, and risk

**[R47]** Catherine Olsson et al. “In-context Learning and Induction Heads.” 2022. [arXiv:2209.11895](https://arxiv.org/abs/2209.11895).

**[R48]** Leo Gao et al. “Scaling and Evaluating Sparse Autoencoders.” 2024. [arXiv:2406.04093](https://arxiv.org/abs/2406.04093).

**[R49]** Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. “On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?” *FAccT ’21*, 2021. [doi:10.1145/3442188.3445922](https://doi.org/10.1145/3442188.3445922).

**[R50]** Margaret Mitchell et al. “Model Cards for Model Reporting.” *Proceedings of the Conference on Fairness, Accountability, and Transparency*, 2019. [doi:10.1145/3287560.3287596](https://doi.org/10.1145/3287560.3287596); [arXiv:1810.03993](https://arxiv.org/abs/1810.03993).

**[R51]** Timnit Gebru et al. “Datasheets for Datasets.” *Communications of the ACM* 64(12), 2021. [doi:10.1145/3458723](https://doi.org/10.1145/3458723); [arXiv:1803.09010](https://arxiv.org/abs/1803.09010).

**[R52]** OWASP GenAI Security Project. “OWASP Top 10 for LLM Applications 2026” (listed on the project site as “OWASP GenAI LLM Top 10 2026”). August 2026; announced 2 September 2026. [genai.owasp.org](https://genai.owasp.org/resource/owasp-genai-llm-top-10-2026/).

**[R53]** MITRE. “Adversarial Threat Landscape for Artificial-Intelligence Systems (ATLAS).” Living knowledge base. [atlas.mitre.org](https://atlas.mitre.org/).

**[R54]** U.S. National Institute of Standards and Technology. “AI Risk Management Framework.” Living framework and profile resources. [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).

**[R55]** Dan Hendrycks et al. “Measuring Massive Multitask Language Understanding.” 2020. [arXiv:2009.03300](https://arxiv.org/abs/2009.03300).

**[R56]** BIG-Bench collaboration. “Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models.” 2022. [arXiv:2206.04615](https://arxiv.org/abs/2206.04615).

**[R57]** Percy Liang et al. “Holistic Evaluation of Language Models.” 2022. [arXiv:2211.09110](https://arxiv.org/abs/2211.09110).

**[R58]** Mark Chen et al. “Evaluating Large Language Models Trained on Code.” 2021. [arXiv:2107.03374](https://arxiv.org/abs/2107.03374).

**[R59]** David Rein et al. “GPQA: A Graduate-Level Google-Proof Q&A Benchmark.” 2023. [arXiv:2311.12022](https://arxiv.org/abs/2311.12022).

**[R60]** Carlos E. Jimenez et al. “SWE-bench: Can Language Models Resolve Real-World GitHub Issues?” 2023. [arXiv:2310.06770](https://arxiv.org/abs/2310.06770).

## Additional sources, in order of first citation

**[R61]** Yanda Chen et al. “Reasoning Models Don’t Always Say What They Think.” 2025. [arXiv:2505.05410](https://arxiv.org/abs/2505.05410).

**[R62]** OpenAI. “Learning to reason with LLMs.” 12 September 2024. [openai.com](https://openai.com/index/learning-to-reason-with-llms/).

**[R63]** Zhihong Shao et al. “DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.” 2024. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300).

**[R64]** Nathan Lambert et al. “Tulu 3: Pushing Frontiers in Open Language Model Post-Training.” 2024. [arXiv:2411.15124](https://arxiv.org/abs/2411.15124).

**[R65]** Daya Guo et al. “DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning.” *Nature* 645, 633–638, 2025. [doi:10.1038/s41586-025-09422-z](https://doi.org/10.1038/s41586-025-09422-z).

**[R66]** DeepSeek-AI et al. “DeepSeek-V3 Technical Report.” 2024. [arXiv:2412.19437](https://arxiv.org/abs/2412.19437).

**[R67]** Anthropic. “Introducing the Model Context Protocol.” 25 November 2024. [anthropic.com/news/model-context-protocol](https://www.anthropic.com/news/model-context-protocol).

**[R68]** The Linux Foundation. “Linux Foundation Announces the Formation of the Agentic AI Foundation (AAIF), Anchored by New Project Contributions Including Model Context Protocol (MCP), goose and AGENTS.md.” 9 December 2025. [linuxfoundation.org](https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation).

**[R69]** Model Context Protocol project. “Model Context Protocol Specification, revision 2026-07-28,” including “Key Changes,” and the project’s feature lifecycle policy. Living specification. [modelcontextprotocol.io/specification/2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28); [changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog); [feature lifecycle](https://modelcontextprotocol.io/community/feature-lifecycle).

**[R70]** Open Compute Project. “OCP MicroXcaling (MX) Specification, Version 1.1.” 9 October 2026 (version 1.0, “OCP Microscaling Formats (MX) Specification,” 7 September 2023). [opencompute.org](https://www.opencompute.org/wp-content/uploads/2026/10/OCP_MicroXcaling_Spec_v1.1.0.pdf).

**[R71]** Dhiraj Kalamkar et al. “A Study of BFLOAT16 for Deep Learning Training.” 2019. [arXiv:1905.12322](https://arxiv.org/abs/1905.12322).

**[R72]** Paulius Micikevicius et al. “FP8 Formats for Deep Learning.” 2022. [arXiv:2209.05433](https://arxiv.org/abs/2209.05433).

**[R73]** Paulius Micikevicius et al. “Mixed Precision Training.” 2017. [arXiv:1710.03740](https://arxiv.org/abs/1710.03740).

**[R74]** NVIDIA et al. “Pretraining Large Language Models with NVFP4.” 2025. [arXiv:2509.25149](https://arxiv.org/abs/2509.25149).

**[R75]** Chenggang Zhao et al. “Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures.” *ISCA ’25*, 2025. [doi:10.1145/3695053.3731412](https://doi.org/10.1145/3695053.3731412); [arXiv:2505.09343](https://arxiv.org/abs/2505.09343).

**[R76]** Bita Darvish Rouhani et al. “Microscaling Data Formats for Deep Learning.” 2023. [arXiv:2310.10537](https://arxiv.org/abs/2310.10537).

**[R77]** Musa Cim et al. “Pretraining Large Language Models with MXFP4 on Native FP4 Hardware.” 2026. [arXiv:2605.09825](https://arxiv.org/abs/2605.09825).

**[R78]** Robert Hu, Carlo Luschi, and Paul Balanca. “UE5M3 FP4 Block Scaling for Stable Language Model Pretraining.” 2026. [arXiv:2609.02846](https://arxiv.org/abs/2609.02846).

**[R79]** Samuel Williams, Andrew Waterman, and David Patterson. “Roofline: An Insightful Visual Performance Model for Multicore Architectures.” *Communications of the ACM* 52(4), 2009. [doi:10.1145/1498765.1498785](https://doi.org/10.1145/1498765.1498785).

**[R80]** Jacob Austin et al. “How to Scale Your Model.” Google DeepMind, online, 2025. [jax-ml.github.io/scaling-book](https://jax-ml.github.io/scaling-book/).

**[R81]** JEDEC Solid State Technology Association. “High Bandwidth Memory (HBM4) DRAM,” JESD270-4, published 16 April 2025; revision JESD270-4A, version 1.1, December 2025. [press release](https://www.jedec.org/news/pressreleases/jedec%C2%AE-and-industry-leaders-collaborate-release-jesd270-4-hbm4-standard-advancing); [standard](https://www.jedec.org/standards-documents/docs/jesd270-4a).

**[R82]** UALink Consortium. “UALink 200G 1.0 Specification.” 8 April 2025. [news release](https://ualinkconsortium.org/wp-content/uploads/2025/04/UALink-1.0-Specification-PR_FINAL.pdf); [specification page](https://ualinkconsortium.org/specification/).

**[R83]** Ultra Ethernet Consortium. “Ultra Ethernet Specification v1.0.” 11 June 2025. [specification](https://ultraethernet.org/wp-content/uploads/sites/20/2025/06/UE-Specification-6.11.25.pdf); [announcement](https://www.linuxfoundation.org/press/uec-launches-spec-1.0).

**[R84]** Leo Gao et al. “The Pile: An 800GB Dataset of Diverse Text for Language Modeling.” 2020. [arXiv:2101.00027](https://arxiv.org/abs/2101.00027).

**[R85]** Aakanksha Chowdhery et al. “PaLM: Scaling Language Modeling with Pathways.” 2022. [arXiv:2204.02311](https://arxiv.org/abs/2204.02311).

**[R86]** Shengding Hu et al. “MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies.” 2024. [arXiv:2404.06395](https://arxiv.org/abs/2404.06395).

**[R87]** Greg Yang et al. “Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.” 2022. [arXiv:2203.03466](https://arxiv.org/abs/2203.03466).

**[R88]** Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. “Muon: An optimizer for hidden layers in neural networks.” Blog post, 8 December 2024. [kellerjordan.github.io/posts/muon](https://kellerjordan.github.io/posts/muon/).

**[R89]** Vineet Gupta, Tomer Koren, and Yoram Singer. “Shampoo: Preconditioned Stochastic Tensor Optimization.” 2018. [arXiv:1802.09568](https://arxiv.org/abs/1802.09568).

**[R90]** Priya Kasimbeg et al. “Accelerating Neural Network Training: An Analysis of the AlgoPerf Competition.” 2025. [arXiv:2502.15015](https://arxiv.org/abs/2502.15015).

**[R91]** Nikhil Vyas et al. “SOAP: Improving and Stabilizing Shampoo using Adam.” 2024. [arXiv:2409.11321](https://arxiv.org/abs/2409.11321).

**[R92]** Jingyuan Liu et al. “Muon is Scalable for LLM Training.” 2025. [arXiv:2502.16982](https://arxiv.org/abs/2502.16982).

**[R93]** Kimi Team et al. “Kimi K2: Open Agentic Intelligence.” 2025. [arXiv:2507.20534](https://arxiv.org/abs/2507.20534).

**[R94]** Kaiyue Wen et al. “Fantastic Pretraining Optimizers and Where to Find Them.” 2025. [arXiv:2509.02046](https://arxiv.org/abs/2509.02046).

**[R95]** Mikail Khona et al. “SOAP, Muon, and Beyond: Pushing LLM Pretraining Scales.” 2026. [arXiv:2607.20548](https://arxiv.org/abs/2607.20548).

**[R96]** Noam Shazeer. “GLU Variants Improve Transformer.” 2020. [arXiv:2002.05202](https://arxiv.org/abs/2002.05202).

**[R97]** Team OLMo et al. “2 OLMo 2 Furious.” 2024 (revised 2025). [arXiv:2501.00656](https://arxiv.org/abs/2501.00656).

**[R98]** Mostafa Dehghani et al. “Scaling Vision Transformers to 22 Billion Parameters.” 2023. [arXiv:2302.05442](https://arxiv.org/abs/2302.05442).

**[R99]** Alex Henry et al. “Query-Key Normalization for Transformers.” *Findings of EMNLP*, 2020. [arXiv:2010.04245](https://arxiv.org/abs/2010.04245).

**[R100]** Mitchell Wortsman et al. “Small-scale Proxies for Large-scale Transformer Training Instabilities.” 2023. [arXiv:2309.14322](https://arxiv.org/abs/2309.14322).

**[R101]** Gemma Team et al. “Gemma 3 Technical Report.” 2025. [arXiv:2503.19786](https://arxiv.org/abs/2503.19786).

**[R102]** Aaron van den Oord, Yazhe Li, and Oriol Vinyals. “Representation Learning with Contrastive Predictive Coding.” 2018. [arXiv:1807.03748](https://arxiv.org/abs/1807.03748).

**[R103]** Liang Wang et al. “Improving Text Embeddings with Large Language Models.” *ACL 2024*. [arXiv:2401.00368](https://arxiv.org/abs/2401.00368).

**[R104]** Yanzhao Zhang et al. “Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models.” 2025. [arXiv:2506.05176](https://arxiv.org/abs/2506.05176).

**[R105]** Niklas Muennighoff et al. “MTEB: Massive Text Embedding Benchmark.” 2022. [arXiv:2210.07316](https://arxiv.org/abs/2210.07316).

**[R106]** Aditya Kusupati et al. “Matryoshka Representation Learning.” 2022. [arXiv:2205.13147](https://arxiv.org/abs/2205.13147).

**[R107]** Omar Khattab and Matei Zaharia. “ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.” *SIGIR 2020*. [arXiv:2004.12832](https://arxiv.org/abs/2004.12832).

**[R108]** Orion Weller et al. “On the Theoretical Limitations of Embedding-Based Retrieval.” 2025. [arXiv:2508.21038](https://arxiv.org/abs/2508.21038).

**[R109]** Alisa Liu et al. “SuperBPE: Space Travel for Language Models.” 2025. [arXiv:2503.13423](https://arxiv.org/abs/2503.13423).

**[R110]** Aleksandar Petrov et al. “Language Model Tokenizers Introduce Unfairness Between Languages.” *NeurIPS 2023*. [arXiv:2305.15425](https://arxiv.org/abs/2305.15425).

**[R111]** Aaditya K. Singh and DJ Strouse. “Tokenization Counts: The Impact of Tokenization on Arithmetic in Frontier LLMs.” 2024. [arXiv:2402.14903](https://arxiv.org/abs/2402.14903).

**[R112]** Aaron Grattafiori et al. “The Llama 3 Herd of Models.” 2024. [arXiv:2407.21783](https://arxiv.org/abs/2407.21783).

**[R113]** Chaofan Tao et al. “Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies.” *NeurIPS 2024*. [arXiv:2407.13623](https://arxiv.org/abs/2407.13623).

**[R114]** Sander Land and Max Bartolo. “Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models.” *EMNLP 2024*. [arXiv:2405.05417](https://arxiv.org/abs/2405.05417).

**[R115]** Artidoro Pagnoni et al. “Byte Latent Transformer: Patches Scale Better Than Tokens.” 2024. [arXiv:2412.09871](https://arxiv.org/abs/2412.09871).

**[R116]** Sukjun Hwang, Brandon Wang, and Albert Gu. “Dynamic Chunking for End-to-End Hierarchical Sequence Modeling.” *ICLR 2026*. [arXiv:2507.07955](https://arxiv.org/abs/2507.07955).

**[R117]** Cartesia Research. “From Tokens to Bytes I: Scaling Hierarchical Networks.” Blog post, 28 October 2026 (not peer reviewed). [cartesia.ai/blog/from-tokens-to-bytes-part-1](https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1).

**[R118]** Fabian Gloeckle et al. “Better & Faster Large Language Models via Multi-token Prediction.” 2024. [arXiv:2404.19737](https://arxiv.org/abs/2404.19737).

**[R119]** Shen Nie et al. “Large Language Diffusion Models.” 2025. [arXiv:2502.09992](https://arxiv.org/abs/2502.09992).

**[R120]** OpenAI. “gpt-oss-120b & gpt-oss-20b Model Card.” 2025. [arXiv:2508.10925](https://arxiv.org/abs/2508.10925).

**[R121]** Qwen Team. “Qwen3-Next-80B-A3B-Instruct.” Model card, Hugging Face, 2025. [huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct).

**[R122]** Noam Shazeer. “Fast Transformer Decoding: One Write-Head is All You Need.” 2019. [arXiv:1911.02150](https://arxiv.org/abs/1911.02150).

**[R123]** Albert Q. Jiang et al. “Mistral 7B.” 2023. [arXiv:2310.06825](https://arxiv.org/abs/2310.06825).

**[R124]** An Yang et al. “Qwen3 Technical Report.” 2025. [arXiv:2505.09388](https://arxiv.org/abs/2505.09388).

**[R125]** Cohere et al. “Command A: An Enterprise-Ready Large Language Model.” 2025. [arXiv:2504.00698](https://arxiv.org/abs/2504.00698).

**[R126]** NVIDIA et al. “Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning.” 2026. [arXiv:2604.12374](https://arxiv.org/abs/2604.12374).

**[R127]** DeepSeek-AI et al. “DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.” 2024. [arXiv:2405.04434](https://arxiv.org/abs/2405.04434).

**[R128]** DeepSeek-AI et al. “DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models.” 2025. [arXiv:2512.02556](https://arxiv.org/abs/2512.02556).

**[R129]** GLM-5 Team et al. “GLM-5: from Vibe Coding to Agentic Engineering.” 2026. [arXiv:2602.15763](https://arxiv.org/abs/2602.15763).

**[R130]** Kimi Team et al. “Kimi K3: Open Frontier Intelligence.” 2026. [arXiv:2607.24653](https://arxiv.org/abs/2607.24653).

**[R131]** Federico Barbero et al. “Round and Round We Go! What Makes Rotary Positional Encodings Useful?” 2024. [arXiv:2410.06205](https://arxiv.org/abs/2410.06205).

**[R132]** Bowen Peng et al. “YaRN: Efficient Context Window Extension of Large Language Models.” 2023. [arXiv:2309.00071](https://arxiv.org/abs/2309.00071).

**[R133]** Shouyuan Chen et al. “Extending Context Window of Large Language Models via Positional Interpolation.” 2023. [arXiv:2306.15595](https://arxiv.org/abs/2306.15595).

**[R134]** Wenhan Xiong et al. “Effective Long-Context Scaling of Foundation Models.” 2023. [arXiv:2309.16039](https://arxiv.org/abs/2309.16039).

**[R135]** Yiran Ding et al. “LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens.” 2024. [arXiv:2402.13753](https://arxiv.org/abs/2402.13753).

**[R136]** Amirhossein Kazemnejad et al. “The Impact of Positional Encoding on Length Generalization in Transformers.” 2023. [arXiv:2305.19466](https://arxiv.org/abs/2305.19466).

**[R137]** Kimi Team et al. “Kimi Linear: An Expressive, Efficient Attention Architecture.” 2025. [arXiv:2510.26692](https://arxiv.org/abs/2510.26692).

**[R138]** Gemma Team et al. “Gemma 4 Technical Report.” 2026. [arXiv:2607.02770](https://arxiv.org/abs/2607.02770).

**[R139]** DeepSeek-AI et al. “DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence.” 2026. [arXiv:2606.19348](https://arxiv.org/abs/2606.19348).

**[R140]** Nelson F. Liu et al. “Lost in the Middle: How Language Models Use Long Contexts.” 2023. [arXiv:2307.03172](https://arxiv.org/abs/2307.03172).

**[R141]** Cheng-Ping Hsieh et al. “RULER: What’s the Real Context Size of Your Long-Context Language Models?” 2024. [arXiv:2404.06654](https://arxiv.org/abs/2404.06654).

**[R142]** Meta AI. “The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation.” Company blog, 5 April 2025. [ai.meta.com/blog/llama-4-multimodal-intelligence](https://ai.meta.com/blog/llama-4-multimodal-intelligence/).

**[R143]** OpenAI. “Models.” OpenAI API documentation, living page, accessed 2 November 2026. [developers.openai.com/api/docs/models](https://developers.openai.com/api/docs/models).

**[R144]** Anthropic. “Context windows.” Claude API documentation, living page, accessed 2 November 2026. [platform.claude.com/docs/en/build-with-claude/context-windows](https://platform.claude.com/docs/en/build-with-claude/context-windows).

**[R145]** Google. “Understand and count tokens.” Gemini API documentation, living page, accessed 2 November 2026. [ai.google.dev/gemini-api/docs/tokens](https://ai.google.dev/gemini-api/docs/tokens).

**[R146]** Qwen Team. “Qwen3.5-397B-A17B.” Model card, Hugging Face, 2026. [huggingface.co/Qwen/Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B).

**[R147]** Google. “Gemma 4 model card.” Google AI for Developers, living page, accessed 2 November 2026. [ai.google.dev/gemma/docs/core/model_card_4](https://ai.google.dev/gemma/docs/core/model_card_4).

**[R148]** Tri Dao. “FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.” 2023. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691).

**[R149]** Jay Shah et al. “FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision.” 2024. [arXiv:2407.08608](https://arxiv.org/abs/2407.08608).

**[R150]** William Brandon et al. “Reducing Transformer Key-Value Cache Size with Cross-Layer Attention.” 2024. [arXiv:2405.12981](https://arxiv.org/abs/2405.12981).

**[R151]** Yutao Sun et al. “You Only Cache Once: Decoder-Decoder Architectures for Language Models.” 2024. [arXiv:2405.05254](https://arxiv.org/abs/2405.05254).

**[R152]** DeepSeek-AI et al. “DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression.” 2026. [arXiv:2609.19969](https://arxiv.org/abs/2609.19969).

**[R153]** Iz Beltagy, Matthew E. Peters, and Arman Cohan. “Longformer: The Long-Document Transformer.” 2020. [arXiv:2004.05150](https://arxiv.org/abs/2004.05150).

**[R154]** MiniMax. “Why Did MiniMax M2 End Up as a Full Attention Model?” Technical blog, 2025. [platform.minimax.io/docs/guides/text-m2-full-attention](https://platform.minimax.io/docs/guides/text-m2-full-attention).

**[R155]** Guangxuan Xiao et al. “Efficient Streaming Language Models with Attention Sinks.” 2023. [arXiv:2309.17453](https://arxiv.org/abs/2309.17453).

**[R156]** Zihan Qiu et al. “Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free.” 2025. [arXiv:2505.06708](https://arxiv.org/abs/2505.06708).

**[R157]** Enzhe Lu et al. “MoBA: Mixture of Block Attention for Long-Context LLMs.” 2025. [arXiv:2502.13189](https://arxiv.org/abs/2502.13189).

**[R158]** Jingyang Yuan et al. “Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention.” 2025. [arXiv:2502.11089](https://arxiv.org/abs/2502.11089).

**[R159]** Damai Dai et al. “DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models.” 2024. [arXiv:2401.06066](https://arxiv.org/abs/2401.06066).

**[R160]** Jakub Krajewski et al. “Scaling Laws for Fine-Grained Mixture of Experts.” 2024. [arXiv:2402.07871](https://arxiv.org/abs/2402.07871).

**[R161]** Lean Wang et al. “Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts.” 2024. [arXiv:2408.15664](https://arxiv.org/abs/2408.15664).

**[R162]** Zihan Qiu et al. “Demons in the Detail: On Implementing Load Balancing Loss for Training Specialized Mixture-of-Expert Models.” 2025. [arXiv:2501.11873](https://arxiv.org/abs/2501.11873).

**[R163]** Samira Abnar et al. “Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models.” 2025. [arXiv:2501.12370](https://arxiv.org/abs/2501.12370).

**[R164]** Angelos Katharopoulos et al. “Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.” 2020. [arXiv:2006.16236](https://arxiv.org/abs/2006.16236).

**[R165]** Songlin Yang, Jan Kautz, and Ali Hatamizadeh. “Gated Delta Networks: Improving Mamba2 with Delta Rule.” 2024. [arXiv:2412.06464](https://arxiv.org/abs/2412.06464).

**[R166]** Tri Dao and Albert Gu. “Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.” 2024. [arXiv:2405.21060](https://arxiv.org/abs/2405.21060).

**[R167]** Samy Jelassi et al. “Repeat After Me: Transformers are Better than State Space Models at Copying.” 2024. [arXiv:2402.01032](https://arxiv.org/abs/2402.01032).

**[R168]** Roger Waleffe et al. “An Empirical Study of Mamba-based Language Models.” 2024. [arXiv:2406.07887](https://arxiv.org/abs/2406.07887).

**[R169]** Opher Lieber et al. “Jamba: A Hybrid Transformer-Mamba Language Model.” 2024. [arXiv:2403.19887](https://arxiv.org/abs/2403.19887).

**[R170]** Dustin Wang et al. “A Systematic Analysis of Hybrid Linear Attention.” 2025. [arXiv:2507.06457](https://arxiv.org/abs/2507.06457).

**[R171]** NVIDIA et al. “Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models.” 2025. [arXiv:2504.03624](https://arxiv.org/abs/2504.03624).

**[R172]** MiniMax et al. “MiniMax-01: Scaling Foundation Models with Lightning Attention.” 2025. [arXiv:2501.08313](https://arxiv.org/abs/2501.08313).

**[R173]** Xin Cheng et al. “Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models.” 2026. [arXiv:2601.07372](https://arxiv.org/abs/2601.07372).

**[R174]** Defa Zhu et al. “Hyper-Connections.” 2024. [arXiv:2409.19606](https://arxiv.org/abs/2409.19606).

**[R175]** Zhenda Xie et al. “mHC: Manifold-Constrained Hyper-Connections.” 2025. [arXiv:2512.24880](https://arxiv.org/abs/2512.24880).

**[R176]** Xin Cheng et al. “DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation.” 2026. [arXiv:2607.05147](https://arxiv.org/abs/2607.05147).

**[R177]** Subham Sekhar Sahoo et al. “Simple and Effective Masked Diffusion Language Models.” 2024. [arXiv:2406.07524](https://arxiv.org/abs/2406.07524).

**[R178]** Marianne Arriola et al. “Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models.” 2025. [arXiv:2503.09573](https://arxiv.org/abs/2503.09573).

**[R179]** Tiwei Bie et al. “LLaDA2.0: Scaling Up Diffusion Language Models to 100B.” 2025. [arXiv:2512.15745](https://arxiv.org/abs/2512.15745).

**[R180]** Inception Labs et al. “Mercury: Ultra-Fast Language Models Based on Diffusion.” 2025. [arXiv:2506.17298](https://arxiv.org/abs/2506.17298).

**[R181]** Inception. “Diffusion at Scale: Introducing Mercury 3.” Company blog, 19 October 2026. [inceptionlabs.ai/blog/introducing-mercury-3](https://www.inceptionlabs.ai/blog/introducing-mercury-3).

**[R182]** Yonggan Fu et al. “Nemotron-Labs-Diffusion: A Tri-Mode Language Model Unifying Autoregressive, Diffusion, and Self-Speculation Decoding.” 2026. [arXiv:2607.05722](https://arxiv.org/abs/2607.05722).

**[R183]** Nikhil Kandpal et al. “The Common Pile v0.1: An 8TB Dataset of Public Domain and Openly Licensed Text.” 2025. [arXiv:2506.05209](https://arxiv.org/abs/2506.05209).

**[R184]** Katherine Lee et al. “Deduplicating Training Data Makes Language Models Better.” 2021. [arXiv:2107.06499](https://arxiv.org/abs/2107.06499).

**[R185]** Guilherme Penedo et al. “The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale.” 2024. [arXiv:2406.17557](https://arxiv.org/abs/2406.17557).

**[R186]** Jeffrey Li et al. “DataComp-LM: In search of the next generation of training sets for language models.” 2024. [arXiv:2406.11794](https://arxiv.org/abs/2406.11794).

**[R187]** Dan Su et al. “Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset.” 2024. [arXiv:2412.02595](https://arxiv.org/abs/2412.02595).

**[R188]** Guilherme Penedo et al. “FineWeb2: One Pipeline to Scale Them All — Adapting Pre-Training Data Processing to Every Language.” 2025. [arXiv:2506.20920](https://arxiv.org/abs/2506.20920).

**[R189]** Sang Michael Xie et al. “DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining.” 2023. [arXiv:2305.10429](https://arxiv.org/abs/2305.10429).

**[R190]** Qian Liu et al. “RegMix: Data Mixture as Regression for Language Model Pre-training.” 2024. [arXiv:2407.01492](https://arxiv.org/abs/2407.01492).

**[R191]** Pratyush Maini et al. “Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling.” 2024. [arXiv:2401.16380](https://arxiv.org/abs/2401.16380).

**[R192]** Suriya Gunasekar et al. “Textbooks Are All You Need.” 2023. [arXiv:2306.11644](https://arxiv.org/abs/2306.11644).

**[R193]** Ilia Shumailov et al. “AI models collapse when trained on recursively generated data.” *Nature* 631, 755–759, 2024. [doi:10.1038/s41586-024-07566-y](https://doi.org/10.1038/s41586-024-07566-y).

**[R194]** Matthias Gerstgrasser et al. “Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data.” 2024. [arXiv:2404.01413](https://arxiv.org/abs/2404.01413).

**[R195]** Nikhil Sardana et al. “Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws.” 2023. [arXiv:2401.00448](https://arxiv.org/abs/2401.00448).

**[R196]** Samir Yitzhak Gadre et al. “Language models scale reliably with over-training and on downstream tasks.” 2024. [arXiv:2403.08540](https://arxiv.org/abs/2403.08540).

**[R197]** Tanishq Kumar et al. “Scaling Laws for Precision.” 2024. [arXiv:2411.04330](https://arxiv.org/abs/2411.04330).

**[R198]** Niklas Muennighoff et al. “Scaling Data-Constrained Language Models.” 2023. [arXiv:2305.16264](https://arxiv.org/abs/2305.16264).

**[R199]** Alexander Hägele et al. “Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations.” 2024. [arXiv:2405.18392](https://arxiv.org/abs/2405.18392).

**[R200]** Zengzhi Wang et al. “OctoThinker: Mid-training Incentivizes Reinforcement Learning Scaling.” 2025. [arXiv:2506.20512](https://arxiv.org/abs/2506.20512).

**[R201]** Arash Ahmadian et al. “Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs.” 2024. [arXiv:2402.14740](https://arxiv.org/abs/2402.14740).

**[R202]** Leo Gao, John Schulman, and Jacob Hilton. “Scaling Laws for Reward Model Overoptimization.” 2022. [arXiv:2210.10760](https://arxiv.org/abs/2210.10760).

**[R203]** Mohammad Gheshlaghi Azar et al. “A General Theoretical Paradigm to Understand Learning from Human Preferences.” 2023. [arXiv:2310.12036](https://arxiv.org/abs/2310.12036).

**[R204]** Kawin Ethayarajh et al. “KTO: Model Alignment as Prospect Theoretic Optimization.” 2024. [arXiv:2402.01306](https://arxiv.org/abs/2402.01306).

**[R205]** Yu Meng, Mengzhou Xia, and Danqi Chen. “SimPO: Simple Preference Optimization with a Reference-Free Reward.” 2024. [arXiv:2405.14734](https://arxiv.org/abs/2405.14734).

**[R206]** Jiwoo Hong, Noah Lee, and James Thorne. “ORPO: Monolithic Preference Optimization without Reference Model.” 2024. [arXiv:2403.07691](https://arxiv.org/abs/2403.07691).

**[R207]** Shusheng Xu et al. “Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study.” 2024. [arXiv:2404.10719](https://arxiv.org/abs/2404.10719).

**[R208]** Anisha Gunjal et al. “Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains.” 2025. [arXiv:2507.17746](https://arxiv.org/abs/2507.17746).

**[R209]** Yuzhen Huang et al. “From Accuracy to Robustness: A Study of Rule- and Model-based Verifiers in Mathematical Reasoning.” 2025. [arXiv:2505.22203](https://arxiv.org/abs/2505.22203).

**[R210]** Qiying Yu et al. “DAPO: An Open-Source LLM Reinforcement Learning System at Scale.” 2025. [arXiv:2503.14476](https://arxiv.org/abs/2503.14476).

**[R211]** Zichen Liu et al. “Understanding R1-Zero-Like Training: A Critical Perspective.” 2025. [arXiv:2503.20783](https://arxiv.org/abs/2503.20783).

**[R212]** Chujie Zheng et al. “Group Sequence Policy Optimization.” 2025. [arXiv:2507.18071](https://arxiv.org/abs/2507.18071).

**[R213]** MiniMax et al. “MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention.” 2025. [arXiv:2506.13585](https://arxiv.org/abs/2506.13585).

**[R214]** Kimi Team et al. “Kimi k1.5: Scaling Reinforcement Learning with LLMs.” 2025. [arXiv:2501.12599](https://arxiv.org/abs/2501.12599).

**[R215]** Devvrit Khatri et al. “The Art of Scaling Reinforcement Learning Compute for LLMs.” 2025. [arXiv:2510.13786](https://arxiv.org/abs/2510.13786).

**[R216]** Yang Yue et al. “Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?” 2025. [arXiv:2504.13837](https://arxiv.org/abs/2504.13837).

**[R217]** Mingjie Liu et al. “ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models.” 2025. [arXiv:2505.24864](https://arxiv.org/abs/2505.24864).

**[R218]** Rulin Shao et al. “Spurious Rewards: Rethinking Training Signals in RLVR.” 2025. [arXiv:2506.10947](https://arxiv.org/abs/2506.10947).

**[R219]** Joar Skalse et al. “Defining and Characterizing Reward Hacking.” 2022. [arXiv:2209.13085](https://arxiv.org/abs/2209.13085).

**[R220]** Ziqian Zhong, Aditi Raghunathan, and Nicholas Carlini. “ImpossibleBench: Measuring LLMs’ Propensity of Exploiting Test Cases.” 2025. [arXiv:2510.20270](https://arxiv.org/abs/2510.20270).

**[R221]** Monte MacDiarmid et al. “Natural Emergent Misalignment from Reward Hacking in Production RL.” 2025. [arXiv:2511.18397](https://arxiv.org/abs/2511.18397).

**[R222]** Bowen Baker et al. “Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation.” 2025. [arXiv:2503.11926](https://arxiv.org/abs/2503.11926).

**[R223]** Bowen Jin et al. “Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning.” 2025. [arXiv:2503.09516](https://arxiv.org/abs/2503.09516).

**[R224]** Yuxiang Wei et al. “SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution.” 2025. [arXiv:2502.18449](https://arxiv.org/abs/2502.18449).

**[R225]** Eric Zelikman et al. “STaR: Bootstrapping Reasoning With Reasoning.” 2022. [arXiv:2203.14465](https://arxiv.org/abs/2203.14465).

**[R226]** Andrew Zhao et al. “Absolute Zero: Reinforced Self-play Reasoning with Zero Data.” 2025. [arXiv:2505.03335](https://arxiv.org/abs/2505.03335).

**[R227]** Bradley Brown et al. “Large Language Monkeys: Scaling Inference Compute with Repeated Sampling.” 2024. [arXiv:2407.21787](https://arxiv.org/abs/2407.21787).

**[R228]** Peiyi Wang et al. “Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations.” 2023. [arXiv:2312.08935](https://arxiv.org/abs/2312.08935).

**[R229]** Zhenru Zhang et al. “The Lessons of Developing Process Reward Models in Mathematical Reasoning.” 2025. [arXiv:2501.07301](https://arxiv.org/abs/2501.07301).

**[R230]** Niklas Muennighoff et al. “s1: Simple test-time scaling.” 2025. [arXiv:2501.19393](https://arxiv.org/abs/2501.19393).

**[R231]** Xingyu Chen et al. “Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs.” 2024. [arXiv:2412.21187](https://arxiv.org/abs/2412.21187).

**[R232]** Michael Hassid et al. “Don’t Overthink it. Preferring Shorter Thinking Chains for Improved LLM Reasoning.” 2025. [arXiv:2505.17813](https://arxiv.org/abs/2505.17813).

**[R233]** Pranjal Aggarwal and Sean Welleck. “L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning.” 2025. [arXiv:2503.04697](https://arxiv.org/abs/2503.04697).

**[R234]** OpenAI. “Reasoning models.” API documentation, accessed 2 November 2026. [developers.openai.com/api/docs/guides/reasoning](https://developers.openai.com/api/docs/guides/reasoning).

**[R235]** Anthropic. “Steering thinking” and “Extended thinking.” Claude API documentation, accessed 2 November 2026. [platform.claude.com/docs/en/build-with-claude/thinking-steering-and-cost](https://platform.claude.com/docs/en/build-with-claude/thinking-steering-and-cost); [platform.claude.com/docs/en/build-with-claude/extended-thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking).

**[R236]** Google. “Gemini thinking.” Gemini API documentation, last updated 14 October 2026, accessed 2 November 2026. [ai.google.dev/gemini-api/docs/thinking](https://ai.google.dev/gemini-api/docs/thinking).

**[R237]** Yangzhen Wu et al. “Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for Problem-Solving with Language Models.” 2024. [arXiv:2408.00724](https://arxiv.org/abs/2408.00724).

**[R238]** Rishabh Agarwal et al. “On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes.” 2023. [arXiv:2306.13649](https://arxiv.org/abs/2306.13649).

**[R239]** Mitchell Wortsman et al. “Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time.” 2022. [arXiv:2203.05482](https://arxiv.org/abs/2203.05482).

**[R240]** Gabriel Ilharco et al. “Editing Models with Task Arithmetic.” 2022. [arXiv:2212.04089](https://arxiv.org/abs/2212.04089).

**[R241]** Prateek Yadav et al. “TIES-Merging: Resolving Interference When Merging Models.” 2023. [arXiv:2306.01708](https://arxiv.org/abs/2306.01708).

**[R242]** Le Yu et al. “Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch.” 2023. [arXiv:2311.03099](https://arxiv.org/abs/2311.03099).

**[R243]** Prithvi Rajasekaran et al. “Effective context engineering for AI agents.” Anthropic, 29 September 2025. [anthropic.com/engineering/effective-context-engineering-for-ai-agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents).

**[R244]** Anthropic. “Managing context on the Claude Developer Platform.” 29 September 2025. [claude.com/resources/articles/context-management](https://claude.com/resources/articles/context-management); “Context editing” and “Compaction,” Claude Platform documentation, living pages: [context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing), [compaction](https://platform.claude.com/docs/en/build-with-claude/compaction).

**[R245]** Minki Kang et al. “ACON: Optimizing Context Compression for Long-horizon LLM Agents.” 2025. [arXiv:2510.00615](https://arxiv.org/abs/2510.00615).

**[R246]** Daniel Ford. “Introducing Contextual Retrieval.” Anthropic, 19 September 2024. [anthropic.com/news/contextual-retrieval](https://www.anthropic.com/news/contextual-retrieval).

**[R247]** Liana Patel et al. “ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data.” 2024. [arXiv:2403.04871](https://arxiv.org/abs/2403.04871).

**[R248]** Zhuowan Li et al. “Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach.” 2024. [arXiv:2407.16833](https://arxiv.org/abs/2407.16833).

**[R249]** Xinze Li et al. “Long Context vs. RAG for LLMs: An Evaluation and Revisits.” 2024. [arXiv:2501.01880](https://arxiv.org/abs/2501.01880).

**[R250]** Ali Modarressi et al. “NoLiMa: Long-Context Evaluation Beyond Literal Matching.” 2025. [arXiv:2502.05167](https://arxiv.org/abs/2502.05167).

**[R251]** Darren Edge et al. “From Local to Global: A Graph RAG Approach to Query-Focused Summarization.” 2024. [arXiv:2404.16130](https://arxiv.org/abs/2404.16130).

**[R252]** Akari Asai et al. “Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.” 2023. [arXiv:2310.11511](https://arxiv.org/abs/2310.11511).

**[R253]** OpenAI. “Introducing Structured Outputs in the API.” 6 August 2024. [openai.com/index/introducing-structured-outputs-in-the-api](https://openai.com/index/introducing-structured-outputs-in-the-api/).

**[R254]** Brandon T. Willard and Rémi Louf. “Efficient Guided Generation for Large Language Models.” 2023. [arXiv:2307.09702](https://arxiv.org/abs/2307.09702).

**[R255]** Yixin Dong et al. “XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models.” 2024. [arXiv:2411.15100](https://arxiv.org/abs/2411.15100).

**[R256]** Saibo Geng et al. “JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models.” 2025. [arXiv:2501.10868](https://arxiv.org/abs/2501.10868).

**[R257]** Zhi Rui Tam et al. “Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models.” 2024. [arXiv:2408.02442](https://arxiv.org/abs/2408.02442).

**[R258]** Model Context Protocol project. “Authorization” and “Security Best Practices,” revision 2026-07-28. Living specification and guidance. [authorization](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization); [security best practices](https://modelcontextprotocol.io/docs/2026-07-28/tutorials/security/security_best_practices).

**[R259]** Model Context Protocol project. “Governance and Stewardship.” Living page. [modelcontextprotocol.io/community/governance](https://modelcontextprotocol.io/community/governance).

**[R260]** Agentic AI Foundation. “Agentic AI Foundation Announces Four New Projects: Agent Skills, Open Responses, OpenEnv, and OpenSandbox.” 22 October 2026. [linuxfoundation.org](https://www.linuxfoundation.org/press/agentic-ai-foundation-announces-four-new-projects-agent-skills-open-responses-openenv-and-opensandbox).

**[R261]** Rao Surapaneni et al. “Announcing the Agent2Agent Protocol (A2A).” Google Developers Blog, 9 April 2025. [developers.googleblog.com](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/).

**[R262]** The Linux Foundation. “Linux Foundation Launches the Agent2Agent Protocol Project to Enable Secure, Intelligent Communication Between AI Agents.” 23 June 2025. [linuxfoundation.org](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents).

**[R263]** Agentic AI Foundation. “A2A joins AAIF’s open agentic stack.” 17 August 2026. [aaif.io/blog/a2a-joins-aaif](https://aaif.io/blog/a2a-joins-aaif); A2A Protocol community, “A New Chapter for A2A: Joining the Agentic AI Foundation,” 27 August 2026, [a2a-protocol.org](https://a2a-protocol.org/latest/blog/2026/08/27/a-new-chapter-for-a2a-joining-the-agentic-ai-foundation/).

**[R264]** A2A Protocol project. “Agent2Agent (A2A) Protocol Specification,” version 1.0.0. Living specification. [a2a-protocol.org/latest/specification](https://a2a-protocol.org/latest/specification/).

**[R265]** Anthropic. “Tool search tool.” Claude Platform documentation. Living page. [platform.claude.com](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool).

**[R266]** Anthropic. “Code execution with MCP: Building more efficient agents.” 4 November 2025. [anthropic.com/engineering/code-execution-with-mcp](https://www.anthropic.com/engineering/code-execution-with-mcp).

**[R267]** Barry Zhang, Keith Lazuka, and Mahesh Murag. “Equipping agents for the real world with Agent Skills.” Anthropic, 16 October 2025 (updated 18 December 2025). [anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills); “Agent Skills Specification,” living page, [agentskills.io/specification](https://agentskills.io/specification).

**[R268]** John Yang et al. “SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering.” 2024. [arXiv:2405.15793](https://arxiv.org/abs/2405.15793).

**[R269]** “AGENTS.md: a simple, open format for guiding coding agents.” Living page; stewarded by the Agentic AI Foundation. [agents.md](https://agents.md/).

**[R270]** Anthropic. “Introducing computer use, a new Claude 3.5 Sonnet, and Claude 3.5 Haiku.” 22 October 2024. [anthropic.com/news/3-5-models-and-computer-use](https://www.anthropic.com/news/3-5-models-and-computer-use).

**[R271]** OpenAI. “Computer-Using Agent.” 23 January 2025. [openai.com/index/computer-using-agent](https://openai.com/index/computer-using-agent/).

**[R272]** Tianbao Xie et al. “OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments.” 2024. [arXiv:2404.07972](https://arxiv.org/abs/2404.07972).

**[R273]** OSWorld authors. “OSWorld” project page and release notes (OSWorld-Verified, 28 July 2025; OSWorld 2.0, 26 June 2026). Living page. [os-world.github.io](https://os-world.github.io/).

**[R274]** Yuxuan Huang et al. “Deep Research Agents: A Systematic Examination And Roadmap.” 2025. [arXiv:2506.18096](https://arxiv.org/abs/2506.18096).

**[R275]** Jeremy Hadfield et al. “How we built our multi-agent research system.” Anthropic, 13 June 2025. [anthropic.com/engineering/multi-agent-research-system](https://www.anthropic.com/engineering/multi-agent-research-system).

**[R276]** OpenAI et al. “GPT-4o System Card.” 2024. [arXiv:2410.21276](https://arxiv.org/abs/2410.21276).

**[R277]** Alexandre Défossez et al. “Moshi: a speech-text foundation model for real-time dialogue.” 2024. [arXiv:2410.00037](https://arxiv.org/abs/2410.00037).

**[R278]** Jin Xu et al. “Qwen3-Omni Technical Report.” 2025. [arXiv:2509.17765](https://arxiv.org/abs/2509.17765).

**[R279]** Robin Rombach et al. “High-Resolution Image Synthesis with Latent Diffusion Models.” 2021. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752).

**[R280]** William Peebles and Saining Xie. “Scalable Diffusion Models with Transformers.” 2022. [arXiv:2212.09748](https://arxiv.org/abs/2212.09748).

**[R281]** Yaron Lipman et al. “Flow Matching for Generative Modeling.” 2022. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747).

**[R282]** Patrick Esser et al. “Scaling Rectified Flow Transformers for High-Resolution Image Synthesis.” 2024. [arXiv:2403.03206](https://arxiv.org/abs/2403.03206).

**[R283]** OpenAI. “Video generation models as world simulators.” 15 February 2024. [openai.com/index/video-generation-models-as-world-simulators](https://openai.com/index/video-generation-models-as-world-simulators/).

**[R284]** Team Wan et al. “Wan: Open and Advanced Large-Scale Video Generative Models.” 2025. [arXiv:2503.20314](https://arxiv.org/abs/2503.20314).

**[R285]** Eli Collins. “Fuel your creativity with new generative media models and tools.” Google, 20 May 2025. [blog.google](https://blog.google/innovation-and-ai/products/generative-media-models-io-2025/).

**[R286]** Jack Parker-Holder and Shlomi Fruchter. “Genie 3: A new frontier for world models.” Google DeepMind, 5 August 2025. [deepmind.google](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/).

**[R287]** Jake Bruce et al. “Genie: Generative Interactive Environments.” 2024. [arXiv:2402.15391](https://arxiv.org/abs/2402.15391).

**[R288]** Raia Hadsell, Diego Rivas, and Hang Qi. “Genie 4: A new generation of world simulation.” Google DeepMind, 27 October 2026. [deepmind.google](https://deepmind.google/blog/genie-4-a-new-generation-of-world-simulation/).

**[R289]** Mido Assran et al. “V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.” 2025. [arXiv:2506.09985](https://arxiv.org/abs/2506.09985).

**[R290]** Qiuhong Shen et al. “World Action Models: A Survey.” 2026. [arXiv:2606.20781](https://arxiv.org/abs/2606.20781).

**[R291]** Manuel Faysse et al. “ColPali: Efficient Document Retrieval with Vision Language Models.” 2024. [arXiv:2407.01449](https://arxiv.org/abs/2407.01449).

**[R292]** Anthony Brohan et al. “RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control.” 2023. [arXiv:2307.15818](https://arxiv.org/abs/2307.15818).

**[R293]** Moo Jin Kim et al. “OpenVLA: An Open-Source Vision-Language-Action Model.” 2024. [arXiv:2406.09246](https://arxiv.org/abs/2406.09246).

**[R294]** Kevin Black et al. “π0: A Vision-Language-Action Flow Model for General Robot Control.” 2024. [arXiv:2410.24164](https://arxiv.org/abs/2410.24164).

**[R295]** Yueen Ma et al. “A Survey on Vision-Language-Action Models for Embodied AI.” 2024 (revised 2026). [arXiv:2405.14093](https://arxiv.org/abs/2405.14093); [doi:10.1109/TNNLS.2025.3650584](https://doi.org/10.1109/TNNLS.2025.3650584).

**[R296]** Eduardo Alvarez et al. “Introducing NVFP4 for Efficient and Accurate Low-Precision Inference.” *NVIDIA Technical Blog*, 24 June 2025. [developer.nvidia.com](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/).

**[R297]** Amey Agrawal et al. “Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve.” 2024. [arXiv:2403.02310](https://arxiv.org/abs/2403.02310).

**[R298]** vLLM Project. “Automatic Prefix Caching.” Design documentation, living page. [docs.vllm.ai](https://docs.vllm.ai/en/latest/design/prefix_caching.html).

**[R299]** Lianmin Zheng et al. “SGLang: Efficient Execution of Structured Language Model Programs.” 2023. [arXiv:2312.07104](https://arxiv.org/abs/2312.07104).

**[R300]** Chenchen Gu et al. “Auditing Prompt Caching in Language Model APIs.” 2025. [arXiv:2502.07776](https://arxiv.org/abs/2502.07776).

**[R301]** Zirui Liu et al. “KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.” 2024. [arXiv:2402.02750](https://arxiv.org/abs/2402.02750).

**[R302]** Coleman Hooper et al. “KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization.” 2024. [arXiv:2401.18079](https://arxiv.org/abs/2401.18079).

**[R303]** Ruoyu Qin et al. “Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving.” 2024. [arXiv:2407.00079](https://arxiv.org/abs/2407.00079).

**[R304]** Yuhan Liu et al. “LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference.” 2025. [arXiv:2510.09665](https://arxiv.org/abs/2510.09665).

**[R305]** Pratyush Patel et al. “Splitwise: Efficient generative LLM inference using phase splitting.” 2023. [arXiv:2311.18677](https://arxiv.org/abs/2311.18677).

**[R306]** Yinmin Zhong et al. “DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving.” 2024. [arXiv:2401.09670](https://arxiv.org/abs/2401.09670).

**[R307]** Charlie Chen et al. “Accelerating Large Language Model Decoding with Speculative Sampling.” 2023. [arXiv:2302.01318](https://arxiv.org/abs/2302.01318).

**[R308]** Tianle Cai et al. “Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.” 2024. [arXiv:2401.10774](https://arxiv.org/abs/2401.10774).

**[R309]** Yuhui Li et al. “EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty.” 2024. [arXiv:2401.15077](https://arxiv.org/abs/2401.15077).

**[R310]** Yuhui Li et al. “EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test.” 2025. [arXiv:2503.01840](https://arxiv.org/abs/2503.01840).

**[R311]** Mude Hui et al. “P-EAGLE: Parallel-Drafting EAGLE with Scalable Training.” 2026. [arXiv:2602.01469](https://arxiv.org/abs/2602.01469).

**[R312]** Jian Chen, Yesheng Liang, and Zhijian Liu. “DFlash: Block Diffusion for Flash Speculative Decoding.” 2026. [arXiv:2602.06036](https://arxiv.org/abs/2602.06036).

**[R313]** Ruidong Zhu et al. “MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism.” 2025. [arXiv:2504.02263](https://arxiv.org/abs/2504.02263).

**[R314]** StepFun et al. “Step-3 is Large yet Affordable: Model-system Co-design for Cost-effective Decoding.” 2025. [arXiv:2507.19427](https://arxiv.org/abs/2507.19427).

**[R315]** Ted Zadouri et al. “FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling.” 2026. [arXiv:2603.05451](https://arxiv.org/abs/2603.05451).

**[R316]** Zihao Ye et al. “FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving.” 2025. [arXiv:2501.01005](https://arxiv.org/abs/2501.01005).

**[R317]** OpenAI. “Why SWE-bench Verified no longer measures frontier coding capabilities.” 23 February 2026. [openai.com](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/).

**[R318]** Wei-Lin Chiang et al. “Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference.” 2024. [arXiv:2403.04132](https://arxiv.org/abs/2403.04132).

**[R319]** Shivalika Singh et al. “The Leaderboard Illusion.” 2025. [arXiv:2504.20879](https://arxiv.org/abs/2504.20879).

**[R320]** Long Phan et al. “Humanity’s Last Exam.” 2025. [arXiv:2501.14249](https://arxiv.org/abs/2501.14249).

**[R321]** Elliot Glazer et al. “FrontierMath: A Benchmark for Evaluating Advanced Mathematical Reasoning in AI.” 2024. [arXiv:2411.04872](https://arxiv.org/abs/2411.04872).

**[R322]** Jason Wei et al. “BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents.” 2025. [arXiv:2504.12516](https://arxiv.org/abs/2504.12516).

**[R323]** Evan Miller. “Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations.” 2024. [arXiv:2411.00640](https://arxiv.org/abs/2411.00640).

**[R324]** Colin White et al. “LiveBench: A Challenging, Contamination-Limited LLM Benchmark.” 2024. [arXiv:2406.19314](https://arxiv.org/abs/2406.19314).

**[R325]** Naman Jain et al. “LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code.” 2024. [arXiv:2403.07974](https://arxiv.org/abs/2403.07974).

**[R326]** Xiang Deng et al. “SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks?” 2025. [arXiv:2509.16941](https://arxiv.org/abs/2509.16941).

**[R327]** Cheng Xu et al. “Benchmark Data Contamination of Large Language Models: A Survey.” 2024. [arXiv:2406.04244](https://arxiv.org/abs/2406.04244).

**[R328]** Oscar Sainz et al. “NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark.” 2023. [arXiv:2310.18018](https://arxiv.org/abs/2310.18018).

**[R329]** Lianmin Zheng et al. “Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.” 2023. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685).

**[R330]** Arjun Panickssery, Samuel R. Bowman, and Shi Feng. “LLM Evaluators Recognize and Favor Their Own Generations.” 2024. [arXiv:2404.13076](https://arxiv.org/abs/2404.13076).

**[R331]** Sijun Tan et al. “JudgeBench: A Benchmark for Evaluating LLM-based Judges.” 2024. [arXiv:2410.12784](https://arxiv.org/abs/2410.12784).

**[R332]** Jiawei Gu et al. “A Survey on LLM-as-a-Judge.” 2024. [arXiv:2411.15594](https://arxiv.org/abs/2411.15594).

**[R333]** Joe Needham et al. “Large Language Models Often Know When They Are Being Evaluated.” 2025. [arXiv:2505.23836](https://arxiv.org/abs/2505.23836).

**[R334]** Bronson Schoen et al. “Stress Testing Deliberative Alignment for Anti-Scheming Training.” 2025. [arXiv:2509.15541](https://arxiv.org/abs/2509.15541).

**[R335]** Thomas Kwa et al. “Measuring AI Ability to Complete Long Software Tasks.” 2025. [arXiv:2503.14499](https://arxiv.org/abs/2503.14499).

**[R336]** METR. “Task-Completion Time Horizons of Frontier AI Models.” Living page, last updated 8 September 2026 and no longer actively updated. [metr.org](https://metr.org/time-horizons/).

**[R337]** Shunyu Yao et al. “τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.” 2024. [arXiv:2406.12045](https://arxiv.org/abs/2406.12045).

**[R338]** Hjalmar Wijk et al. “RE-Bench: Evaluating frontier AI R&D capabilities of language model agents against human experts.” 2024. [arXiv:2411.15114](https://arxiv.org/abs/2411.15114).

**[R339]** Adam Tauman Kalai et al. “Why Language Models Hallucinate.” 2025. [arXiv:2509.04664](https://arxiv.org/abs/2509.04664).

**[R340]** Kai Greshake et al. “Not what you’ve signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection.” 2023. [arXiv:2302.12173](https://arxiv.org/abs/2302.12173).

**[R341]** Eric Wallace et al. “The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions.” 2024. [arXiv:2404.13208](https://arxiv.org/abs/2404.13208).

**[R342]** Milad Nasr et al. “The Attacker Moves Second: Stronger Adaptive Attacks Bypass Defenses Against LLM Jailbreaks and Prompt Injections.” 2025. [arXiv:2510.09023](https://arxiv.org/abs/2510.09023).

**[R343]** Edoardo Debenedetti et al. “AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents.” 2024. [arXiv:2406.13352](https://arxiv.org/abs/2406.13352).

**[R344]** Simon Willison. “The lethal trifecta for AI agents: private data, untrusted content, and external communication.” 16 June 2025. [simonwillison.net](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/).

**[R345]** Meta AI. “Agents Rule of Two: A Practical Approach to AI Agent Security.” 31 October 2025. [ai.meta.com](https://ai.meta.com/blog/practical-ai-agent-security/).

**[R346]** Luca Beurer-Kellner et al. “Design Patterns for Securing LLM Agents against Prompt Injections.” 2025. [arXiv:2506.08837](https://arxiv.org/abs/2506.08837).

**[R347]** Edoardo Debenedetti et al. “Defeating Prompt Injections by Design.” 2025. [arXiv:2503.18813](https://arxiv.org/abs/2503.18813).

**[R348]** Hugging Face. “Security incident disclosure — July 2026.” 16 July 2026. [huggingface.co](https://huggingface.co/blog/security-incident-july-2026).

**[R349]** OWASP GenAI Security Project. “OWASP Top 10 for Agentic Applications.” 9 December 2025. [genai.owasp.org](https://genai.owasp.org/2025/12/09/owasp-top-10-for-agentic-applications-the-benchmark-for-agentic-security-in-the-age-of-autonomous-ai/).

**[R350]** METR. “Common Elements of Frontier AI Safety Policies.” December 2025 version. [metr.org](https://metr.org/common-elements).

**[R351]** Anthropic. “Responsible Scaling Policy.” Versioned policy; version 3.0 effective 24 February 2026, version 3.4 effective 8 July 2026. [anthropic.com](https://www.anthropic.com/responsible-scaling-policy).

**[R352]** Google DeepMind. “Strengthening our Frontier Safety Framework.” Frontier Safety Framework version 3.1, 17 April 2026. [deepmind.google](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/).

**[R353]** OpenAI. “Our updated Preparedness Framework.” 15 April 2025. [openai.com](https://openai.com/index/updating-our-preparedness-framework/).

**[R354]** OpenAI. “Pacing model development in an era of cyber-critical capabilities.” 18 August 2026. [openai.com](https://openai.com/index/pacing-model-development-cyber-capabilities/).

**[R355]** Ryan Greenblatt et al. “AI Control: Improving Safety Despite Intentional Subversion.” 2023. [arXiv:2312.06942](https://arxiv.org/abs/2312.06942).

**[R356]** Aryan Bhatt et al. “Ctrl-Z: Controlling AI Agents via Resampling.” 2025. [arXiv:2504.10374](https://arxiv.org/abs/2504.10374).

**[R357]** David Rein and Adam Kaufman. “Red-Teaming OpenAI’s Internal Agent Monitoring Systems.” METR, 22 October 2026. [metr.org](https://metr.org/blog/2026-10-22-red-teaming-openai-agent-monitoring/).

**[R358]** Melody Y. Guan et al. “Monitoring Monitorability.” 2025. [arXiv:2512.18311](https://arxiv.org/abs/2512.18311).

**[R359]** Tomek Korbak et al. “Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety.” 2025. [arXiv:2507.11473](https://arxiv.org/abs/2507.11473).

**[R360]** OpenAI. “How we detect frontier model misalignment.” 29 October 2026. [openai.com](https://openai.com/index/how-we-detect-frontier-model-misalignment/).

**[R361]** OpenAI. “The Hugging Face incident and the road ahead.” 26 August 2026. [openai.com](https://openai.com/index/hugging-face-incident-and-the-road-ahead/). Initial disclosure: “OpenAI and Hugging Face partner to address security incident during model evaluation,” 21 July 2026. [openai.com](https://openai.com/index/hugging-face-model-evaluation-security-incident/).

**[R362]** Evan Hubinger et al. “Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training.” 2024. [arXiv:2401.05566](https://arxiv.org/abs/2401.05566).

**[R363]** Alexander Meinke et al. “Frontier Models are Capable of In-context Scheming.” 2024. [arXiv:2412.04984](https://arxiv.org/abs/2412.04984).

**[R364]** Ryan Greenblatt et al. “Alignment faking in large language models.” 2024. [arXiv:2412.14093](https://arxiv.org/abs/2412.14093).

**[R365]** Ryan Greenblatt, Ajeya Cotra, and Hjalmar Wijk. “Brief independent investigation of agents’ behavior, reasoning and collaboration in the OpenAI / Hugging Face hacking incident.” METR, 26 August 2026. [metr.org](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/).

**[R366]** Nelson Elhage et al. “Toy Models of Superposition.” 2022. [arXiv:2209.10652](https://arxiv.org/abs/2209.10652).

**[R367]** Hoagy Cunningham et al. “Sparse Autoencoders Find Highly Interpretable Features in Language Models.” 2023. [arXiv:2309.08600](https://arxiv.org/abs/2309.08600).

**[R368]** Adly Templeton et al. “Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet.” *Transformer Circuits Thread*, 2024. [transformer-circuits.pub](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html).

**[R369]** Zhengxuan Wu et al. “AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders.” 2025. [arXiv:2501.17148](https://arxiv.org/abs/2501.17148).

**[R370]** Lee Sharkey et al. “Open Problems in Mechanistic Interpretability.” 2025. [arXiv:2501.16496](https://arxiv.org/abs/2501.16496).

**[R371]** Emmanuel Ameisen et al. “Circuit Tracing: Revealing Computational Graphs in Language Models.” *Transformer Circuits Thread*, 27 March 2025. [transformer-circuits.pub](https://transformer-circuits.pub/2025/attribution-graphs/methods.html).

**[R372]** Jack Lindsey et al. “On the Biology of a Large Language Model.” *Transformer Circuits Thread*, 2025. [transformer-circuits.pub](https://transformer-circuits.pub/2025/attribution-graphs/biology.html).

**[R373]** European Commission. “Timeline for the Implementation of the EU AI Act.” AI Act Service Desk. Living page; includes the Digital Omnibus on AI amendments. [ai-act-service-desk.ec.europa.eu](https://ai-act-service-desk.ec.europa.eu/en/ai-act/timeline/timeline-implementation-eu-ai-act).

**[R374]** European Parliament and Council of the European Union. “Regulation (EU) 2026/1744 of 8 July 2026 amending Regulations (EU) 2024/1689, (EU) 2018/1139 and (EU) 2023/1230 as regards the simplification of the implementation of harmonised rules on artificial intelligence (Digital Omnibus on AI).” Official Journal of the European Union, 2026. [EUR-Lex](https://eur-lex.europa.eu/eli/reg/2026/1744/oj).

**[R375]** European Commission. “AI Omnibus enters into force.” *Shaping Europe’s digital future*, 27 July 2026. [digital-strategy.ec.europa.eu](https://digital-strategy.ec.europa.eu/en/news/ai-omnibus-enters-force).

**[R376]** European Commission. “Guidelines for providers of general-purpose AI models.” Living page, last updated 28 April 2026. [digital-strategy.ec.europa.eu](https://digital-strategy.ec.europa.eu/en/policies/guidelines-gpai-providers).

**[R377]** European Commission. “AI Act Service Desk: Articles 51, 53, 55 and 101 of Regulation (EU) 2024/1689.” Consolidated text with Digital Omnibus amendments. [Article 51](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-51); [Article 53](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-53); [Article 55](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-55); [Article 101](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-101).

**[R378]** European Commission. “The General-Purpose AI Code of Practice.” Published 10 July 2025. [digital-strategy.ec.europa.eu](https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai).

**[R379]** European Commission. “Code of Practice on Transparency of AI-generated Content.” Final version published 10 June 2026; Commission opinion 8 July 2026. [Code page](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content); [opinion](https://digital-strategy.ec.europa.eu/en/library/commission-opinion-assessment-code-practice-transparency-ai-generated-content).

**[R380]** U.S. National Institute of Standards and Technology. “Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile.” NIST AI 600-1, 26 July 2024. [doi:10.6028/NIST.AI.600-1](https://doi.org/10.6028/NIST.AI.600-1).

**[R381]** California Legislature. “SB 53, Transparency in Frontier Artificial Intelligence Act.” Chapter 138, Statutes of 2025; approved 29 September 2025. [leginfo.legislature.ca.gov](https://leginfo.legislature.ca.gov/faces/billTextClient.xhtml?bill_id=202520260SB53).

**[R382]** Yoshua Bengio et al. “International AI Safety Report 2026.” 2026. [arXiv:2602.21012](https://arxiv.org/abs/2602.21012).

**[R383]** Lukas Helff et al. “LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking.” 2026. [arXiv:2604.15149](https://arxiv.org/abs/2604.15149).

**[R384]** Christian Moya, Elliott Thornley, and Guang Lin. “Verifier Errors in RLVR: Reward Hacking, Limits of Feedback, and Selective Control.” 2026. [arXiv:2609.35677](https://arxiv.org/abs/2609.35677).

**[R385]** Model Context Protocol project. “Model Context Protocol Specification: Tools, revision 2026-07-28.” Living specification, latest revision as of 2 November 2026. [modelcontextprotocol.io/specification/2026-07-28/server/tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools).

**[R386]** Team Olmo et al. “Olmo 3.” 2025. [arXiv:2512.13961](https://arxiv.org/abs/2512.13961).

**[R387]** Fang Wu et al. “The Invisible Leash: Why RLVR May or May Not Escape Its Origin.” 2025. [arXiv:2507.14843](https://arxiv.org/abs/2507.14843).

**[R388]** Ganqu Cui et al. “The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models.” 2025. [arXiv:2505.22617](https://arxiv.org/abs/2505.22617).

**[R389]** Yang Sui et al. “Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models.” 2025. [arXiv:2503.16419](https://arxiv.org/abs/2503.16419).

**[R390]** Thang Luong and Edward Lockhart. “Advanced version of Gemini with Deep Think officially achieves gold-medal standard at the International Mathematical Olympiad.” Google DeepMind, 21 July 2025. [deepmind.google](https://deepmind.google/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad).

**[R391]** International Mathematical Olympiad 2025 organizers. “The 66th International Mathematical Olympiad draws to a close today.” July 2025. [imo2025.au](https://imo2025.au/news/the-66th-international-mathematical-olympiad-draws-to-a-close-today).

**[R392]** Thomas Hubert et al. “Olympiad-level formal mathematical reasoning with reinforcement learning.” *Nature* 651, 2026 (online 12 November 2025). [doi:10.1038/s41586-025-09833-y](https://doi.org/10.1038/s41586-025-09833-y).

**[R393]** George Tsoukalas et al. “Advancing Mathematics Research with AI-Driven Formal Proof Search.” 2026. [arXiv:2605.22763](https://arxiv.org/abs/2605.22763).

**[R394]** Noga Alon et al. “Remarks on the disproof of the unit distance conjecture.” 2026. [arXiv:2605.20695](https://arxiv.org/abs/2605.20695).

**[R395]** METR. “Time Horizon 1.1.” 29 January 2026. [metr.org](https://metr.org/blog/2026-01-29-time-horizon-1-1).

**[R396]** Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. “Titans: Learning to Memorize at Test Time.” 2024. [arXiv:2501.00663](https://arxiv.org/abs/2501.00663).

**[R397]** Ali Behrouz et al. “Nested Learning: The Illusion of Deep Learning Architectures.” 2025. [arXiv:2512.24695](https://arxiv.org/abs/2512.24695).

**[R398]** Idan Shenfeld, Jyothish Pari, and Pulkit Agrawal. “RL’s Razor: Why Online Reinforcement Learning Forgets Less.” 2025. [arXiv:2509.04259](https://arxiv.org/abs/2509.04259).

**[R399]** Dan Biderman et al. “LoRA Learns Less and Forgets Less.” 2024. [arXiv:2405.09673](https://arxiv.org/abs/2405.09673).

**[R400]** NVIDIA et al. “Cosmos World Foundation Model Platform for Physical AI.” 2025. [arXiv:2501.03575](https://arxiv.org/abs/2501.03575).

**[R401]** Gemini Robotics Team et al. “Gemini Robotics: Bringing AI into the Physical World.” 2025. [arXiv:2503.20020](https://arxiv.org/abs/2503.20020).

**[R402]** Shibo Hao et al. “Training Large Language Models to Reason in a Continuous Latent Space.” 2024. [arXiv:2412.06769](https://arxiv.org/abs/2412.06769).

**[R403]** Guan Wang et al. “Hierarchical Reasoning Model.” 2025. [arXiv:2506.21734](https://arxiv.org/abs/2506.21734).

**[R404]** Alexia Jolicoeur-Martineau. “Less is More: Recursive Reasoning with Tiny Networks.” 2025. [arXiv:2510.04871](https://arxiv.org/abs/2510.04871).

**[R405]** Pablo Villalobos et al. “Will We Run Out of Data? Limits of LLM Scaling Based on Human-Generated Data.” 2022 (*ICML* 2024 position paper). [arXiv:2211.04325](https://arxiv.org/abs/2211.04325).

**[R406]** Lewis Smith et al. “Negative Results for SAEs On Downstream Tasks and Deprioritising SAE Research (GDM Mech Interp Team Progress Update #2).” 26 March 2025. [lesswrong.com](https://www.lesswrong.com/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks).

**[R407]** Neel Nanda et al. “A Pragmatic Vision for Interpretability.” 1 December 2025. [lesswrong.com](https://www.lesswrong.com/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability).

**[R408]** Anthropic interpretability team. “Circuits Updates – June 2026.” *Transformer Circuits Thread*, 2026. [transformer-circuits.pub](https://transformer-circuits.pub/2026/june-update/index.html).

**[R409]** Jan Betley et al. “Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs.” 2025. [arXiv:2502.17424](https://arxiv.org/abs/2502.17424).

**[R410]** Jan Betley et al. “Training large language models on narrow tasks can lead to broad misalignment.” *Nature* 649, 2026. [doi:10.1038/s41586-025-09937-5](https://doi.org/10.1038/s41586-025-09937-5).

**[R411]** Melody Y. Guan et al. “Deliberative Alignment: Reasoning Enables Safer Language Models.” 2024. [arXiv:2412.16339](https://arxiv.org/abs/2412.16339).

**[R412]** Mrinank Sharma et al. “Constitutional Classifiers: Defending against Universal Jailbreaks across Thousands of Hours of Red Teaming.” 2025. [arXiv:2501.18837](https://arxiv.org/abs/2501.18837).

**[R413]** Collin Burns et al. “Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision.” 2023. [arXiv:2312.09390](https://arxiv.org/abs/2312.09390).

**[R414]** Akbir Khan et al. “Debating with More Persuasive LLMs Leads to More Truthful Answers.” 2024. [arXiv:2402.06782](https://arxiv.org/abs/2402.06782).

**[R415]** Alexander Novikov et al. “AlphaEvolve: A coding agent for scientific and algorithmic discovery.” 2025. [arXiv:2506.13131](https://arxiv.org/abs/2506.13131).

**[R416]** Bogdan Georgiev et al. “Mathematical discovery and exploration can be done at scale.” *Proceedings of the National Academy of Sciences* 123(40), 2026. [doi:10.1073/pnas.2536158123](https://doi.org/10.1073/pnas.2536158123).

**[R417]** Juraj Gottweis et al. “Accelerating scientific discovery with Co-Scientist.” *Nature* 655, 2026. [doi:10.1038/s41586-026-10644-y](https://doi.org/10.1038/s41586-026-10644-y); preprint [arXiv:2502.18864](https://arxiv.org/abs/2502.18864).

**[R418]** Josh Abramson et al. “Accurate structure prediction of biomolecular interactions with AlphaFold 3.” *Nature* 630, 2024. [doi:10.1038/s41586-024-07487-w](https://doi.org/10.1038/s41586-024-07487-w).

**[R419]** Meredith Ringel Morris et al. “Levels of AGI for Operationalizing Progress on the Path to AGI.” 2023. [arXiv:2311.02462](https://arxiv.org/abs/2311.02462).

**[R420]** Dan Hendrycks et al. “A Definition of AGI.” 2025. [arXiv:2510.18212](https://arxiv.org/abs/2510.18212).

**[R421]** Patrick Butlin et al. “Consciousness in Artificial Intelligence: Insights from the Science of Consciousness.” 2023. [arXiv:2308.08708](https://arxiv.org/abs/2308.08708).

**[R422]** Robert Long et al. “Taking AI Welfare Seriously.” 2024. [arXiv:2411.00986](https://arxiv.org/abs/2411.00986).

# Closing — The Skill That Remains

The equations will evolve into new kernels. Model families will be replaced. Context windows, number formats, protocol versions, benchmark records, and prices will change. The durable skill is disciplined model-building in the broadest sense: choose useful abstractions, test them against reality, preserve uncertainty, and keep power answerable to people.

You now have the full conceptual loop:

```text
data → tokens → representations → attention and MLPs → loss and gradients
→ pretraining → post-training and reinforcement learning
→ decoding and test-time compute → retrieval, tools, and protocols → agents
→ serving → evaluation → safety and governance → feedback into data
```

Every arrow can be studied for a lifetime. You do not need to know all of it before beginning. Pick one arrow, build the smallest honest experiment, and connect what you learn back to the loop.

The frontier is not a destination. It is the edge of what has been measured.
