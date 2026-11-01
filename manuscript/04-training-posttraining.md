# Part IV — Creating and Shaping a Language Model

Architecture defines possible computations. Data and optimization decide which computations are learned. Post-training turns a broad continuation model into a controllable assistant, reasoner, tool user, or specialist.

# The Pretraining Data Pipeline

A trillion-token corpus is not one file. It is a changing mixture assembled through collection, extraction, filtering, deduplication, tokenization, and sampling. Data engineering is model engineering.

## Sources

Possible sources include web pages, books, scientific papers, reference works, discussion forums, code repositories, educational material, transcripts, licensed collections, synthetic examples, and organization-specific data. Multimodal models add images, captions, audio, video, sensor streams, and aligned pairs.

Every source raises questions:

- Was collection permitted by law, contract, robots instructions, and social expectation?
- Does the source contain personal, secret, copyrighted, harmful, or security-sensitive material?
- Which languages, regions, classes, ideologies, and genres are overrepresented?
- What metadata and provenance survive processing?
- Can a source later be removed and its influence traced?

“Publicly reachable” is not synonymous with ethically free, factually reliable, or legally unrestricted. Requirements vary by jurisdiction and use.

Openly licensed corpora are one response. The Common Pile v0.1 collects about eight terabytes of public-domain and openly licensed text from 30 sources, and two 7-billion-parameter models trained on 1 and 2 trillion tokens of it were reported competitive with models trained on unlicensed text at similar compute, such as the earlier Llama 1 and 2 at 7 billion parameters [R183]. The study stopped at 7 billion parameters, so it does not show whether the approach matches much larger runs on unlicensed data.

## Extraction and normalization

Raw web content contains navigation, advertisements, cookie banners, scripts, duplicated templates, and malformed encodings. Extraction attempts to preserve meaningful text, document structure, code blocks, tables, and source metadata.

Over-aggressive normalization destroys useful signals such as paragraph boundaries, mathematical notation, indentation, or language-specific characters. Under-cleaning teaches boilerplate and spam. Pipelines should retain enough lineage to audit transformations.

## Deduplication

Exact duplicates overweight repeated documents and encourage memorization. Near-duplicate detection catches mirrors, templates, copied paragraphs, and slightly modified benchmark items.

Common tools include hashes for exact equality, MinHash or locality-sensitive hashing for set similarity, suffix-based matching for long substrings, and embedding similarity for semantic copies. Thresholds trade compute and recall against accidental removal of legitimately repeated phrases.

Deduplication should consider train–test overlap and duplicates *within* evaluation sets. A model answering memorized benchmark text is not evidence of general reasoning.

Deduplication is not only hygiene. In one study, removing near-duplicates made models emit memorized text about ten times less often and reach the same or better accuracy in fewer training steps [R184]. Large pipelines run it in several passes at URL, document, and line level [R112].

Decontamination, which removes evaluation items from training data, uses the same machinery. One published rule drops any text segment that shares an exact 10-gram (ten consecutive units of text) with an evaluation benchmark, and uses exact matching on whole items shorter than ten but at least three units [R63]. Exact-overlap rules miss paraphrases and translations, so they lower contamination risk without eliminating it.

## Quality filtering

Heuristics score language identification, character patterns, length, repetition, profanity, source reputation, formatting, or perplexity under a reference model. Learned classifiers estimate educational value or toxicity. Rules and models can fail systematically.

A quality filter trained on prestigious writing may reject dialect, informal expertise, minority languages, or accessibility-focused text. A toxicity filter may erase documentation of abuse or disproportionately flag reclaimed language. Audit false positives and negatives by slice.

## Model-based quality classifiers

Hand-written heuristics are weak at deciding which pages are worth learning from. Since 2024 the best-documented open pipelines have added a learned classifier in four steps: decide what “good” means by choosing reference examples or writing a rubric; label a sample of documents; train a cheap model on those labels; score the whole corpus and keep documents above a threshold. Published variants differ mainly in the first step:

- **Educational-value rubric.** For FineWeb-Edu, a language model scored 460,000 sampled web pages from 0 to 5 with an additive rubric aimed at grade-school and middle-school knowledge. A linear regression on top of a frozen embedding model, trained on 410,000 of those scores, then rated the rest. Keeping pages that scored at least 3 left 1.3 trillion tokens, and scoring the 15-trillion-token parent corpus took about 6,000 H100 GPU-hours [R185].
- **Resembles good answers.** DataComp-LM compared several model-based filters at small scale. A simple fastText classifier, a bag-of-n-grams model, performed best. The best positive examples were instruction-formatted data and highly rated forum answers, and the final recipe kept the top 10 percent of documents by classifier score [R186].
- **Domain recall loop.** For mathematics, DeepSeekMath trained a fastText classifier on a small seed corpus, used it to recall similar pages from Common Crawl, identified domains where more than 10 percent of pages had been recalled, had people annotate URL patterns in those domains, and retrained. After four rounds the corpus held 120 billion tokens, and the fourth round added little [R63].

The cost of aggressive filtering is quantity. The Nemotron-CC authors note that filters of this kind gave large benchmark gains but removed about 90 percent of the data, which limits very long training runs. They combined an ensemble of classifiers, synthetic rephrasing of documents, and fewer heuristic filters to retain more unique tokens [R187]. Multilingual pipelines face a related problem: FineWeb2 adapts language identification, filtering, and deduplication to each language automatically, was run over more than 1,000 languages, and rebalances data by both duplication count and quality [R188].

Failure modes: a classifier encodes the taste of its reference data, so instruction-style or textbook-style references can pull the corpus toward explanatory prose and thin out other registers, dialects, and languages. Because classifiers are chosen by downstream benchmarks, the filter and the benchmark can co-adapt. Audit by slice, as above, and keep an evaluation that was not used to choose the threshold.

## Privacy and secrets

Pipelines can detect emails, phone numbers, identifiers, credentials, private keys, medical details, and other sensitive patterns. Pattern matching is incomplete; contextual privacy is harder. Memorization risk rises with rarity, duplication, model capacity, and prompting.

Controls include source exclusion, redaction, deduplication, privacy review, data minimization, canary tests, training algorithms with formal privacy guarantees in selected settings, output monitoring, and removal processes. Differential privacy provides a mathematically bounded influence notion but can impose utility and compute costs; it is not achieved by vague “anonymization.”

## Data mixtures

Training does not necessarily sample in raw-source proportion. A mixture assigns weights to domains and languages. Small high-quality datasets may be repeated; huge noisy sources downsampled. Curriculum can change weights over time.

If source `i` has sampling probability `p_i`, its expected token contribution is `p_i × total_tokens`. Repetition can improve a scarce domain but increases overfitting and memorization. Temperature-based sampling flattens source size differences.

Mixture weights are usually tuned with proxy experiments instead of intuition. DoReMi trains a small proxy model (280 million parameters in the paper) with group distributionally robust optimization to produce domain weights, then trains the large model on data resampled with those weights [R189]. RegMix trains many small models on different mixtures, fits a regression that predicts performance from the mixture, and applies the best predicted mixture to a much larger run [R190]. Mixtures also change during a run: the Llama 3 report describes raising the share of non-English and mathematical data and adding more recent web data in later stages, and its reported final mix is roughly 50 percent general knowledge, 25 percent mathematical and reasoning, 17 percent code, and 8 percent multilingual tokens [R112]. A proxy result is evidence, not a guarantee; check that the chosen weights hold at the target scale.

Data mixtures are a form of values and product design. They affect language ability, cultural knowledge, code competence, style, safety, and factuality.

## Synthetic data

Models can generate explanations, exercises, critiques, tool traces, translations, preference pairs, and edge cases. Synthetic data is scalable and controllable but inherits generator errors. Repeated self-training can narrow diversity or amplify artifacts.

Useful safeguards:

- seed from diverse real tasks;
- filter with executable or independent checks;
- use multiple generators and critics;
- preserve hard negatives and uncertainty;
- measure novelty and distribution coverage;
- keep held-out human-authored evaluation sets;
- avoid training and judging entirely with the same model family.

**Synthetic data in pretraining.** What began as small experiments is now routine. Common forms:

- *Rephrasing.* An instruction-tuned model rewrites web documents in a chosen style, such as Wikipedia-like prose or question-and-answer form, and training mixes real and rewritten text. On a noisy web corpus this was reported to speed up pretraining roughly threefold [R191]. Nemotron-CC added several rewriting tasks: diverse question–answer pairs, distilled passages, extracted knowledge, and knowledge lists [R187]. The Kimi K2 report compared three schemes on a factual-question benchmark: repeat a knowledge corpus for ten epochs, rephrase it once and repeat for ten epochs, or rephrase it ten times and train for one pass. Accuracy improved across them in that order [R93].
- *Textbook-style generation.* The phi-1 code model was trained on about 6 billion tokens of web pages selected for “textbook quality” plus about 1 billion tokens of textbooks and exercises generated with GPT-3.5 [R192].
- *Verified generation.* Sample many solutions, keep only those a checker accepts, and train on them. This is the core of rejection sampling and of self-taught reasoning loops (see “Reinforcement Learning for Reasoning and Agents”).

**Model collapse.** Training on a model’s own outputs can degrade the next model. A *Nature* study found that indiscriminate use of model-generated content makes the tails of the original data distribution disappear in recursively trained models [R193]. A follow-up found that this happens when each generation’s synthetic data replaces the real data, while accumulating synthetic data alongside the original real data avoided collapse in its experiments and, in a linear-model analysis, bounded the test error [R194]. The practical rule is to keep real data in the mix and track diversity, not only quality. **Open question:** how much synthetic text frontier pretraining can absorb before diversity or factual grounding suffer.

## Contamination records

A mature pipeline maintains dataset versions, content hashes, source metadata, filtering reasons, licensing categories, and benchmark exclusion lists. When an evaluation leak is discovered, records make diagnosis possible.

::: warning Irreversibility
Once a large run absorbs data, removing a record from the dataset does not automatically remove its influence from the checkpoint. Unlearning is an active research area and needs empirical verification. Prevention and provenance are cheaper than retroactive repair.
:::

# Pretraining at Scale

Pretraining repeatedly presents token sequences, predicts each next token, and adjusts weights. The loop is simple; executing it reliably across thousands of accelerators is not.

## Sequence construction

Documents are tokenized and packed into fixed or variable-length training sequences. Packing reduces padding waste. Boundary tokens and attention masks should prevent unwanted cross-document leakage, depending on the objective.

Long sequences expose long-range dependencies but cost more. A curriculum may begin shorter and later increase length. The position distribution matters: if nearly all evidence is near the beginning, accepting a long context later does not guarantee effective long-range use.

## Objective variants

Common objectives include:

- causal next-token prediction;
- masked-token reconstruction;
- span corruption;
- prefix language modeling;
- fill-in-the-middle for code;
- multi-token prediction, which predicts several future tokens from each position (Part II, “Beyond one next token at a time”);
- multimodal contrastive or generative losses;
- auxiliary router, alignment, or modality losses.

A weighted total can combine objectives:

`L_total = λ₁L_language + λ₂L_image + λ₃L_router + …`

Weights determine gradient influence and may be scheduled.

## Tokens, parameters, and compute

Three scale variables recur:

- `N`: model parameters;
- `D`: training tokens or examples;
- `C`: compute budget.

Empirical scaling laws observed that validation loss often follows approximate power laws plus an irreducible term over studied ranges [R11, R12]. One schematic form is:

`L(N, D) ≈ L∞ + aN^{-α} + bD^{-β}`

Given fixed compute, choosing `N` and `D` is an allocation problem. Chinchilla-style results showed many earlier models were undertrained: smaller models trained on more tokens could outperform larger models at the same training compute [R12]. The headline model had 70 billion parameters and 1.4 trillion training tokens, about 20 tokens per parameter, and “20 tokens per parameter” became a rule of thumb. Treat it as one fitted point, not a constant: the Llama 3 team’s own scaling fit suggested a model of about 402 billion parameters trained on 16.55 trillion tokens, roughly 41 tokens per parameter, for its flagship compute budget [R112].

Scaling laws are fitted empirical models, not promises. They can shift with data quality, architecture, modality, objective, sparsity, optimizer, and evaluation domain. Extrapolation far beyond measurements is uncertain.

## Training past the compute-optimal point

Compute-optimal means cheapest to train for a target loss. It ignores that a deployed model is run many times. A rough lifetime account counts about 6 floating-point operations per parameter per training token (forward and backward passes) and about 2 per parameter per token processed in serving, prompt or generated (forward pass only):

`lifetime FLOPs ≈ 6·N·D_train + 2·N·D_served`

Estimate, with assumptions: take `N` = 8 billion and `D_train` = 15 trillion. Training costs about 7.2 × 10²³ FLOPs. Serving the same 15 trillion tokens costs about 2.4 × 10²³, and serving 45 trillion tokens, three times the training tokens, costs as much as training did. The estimate counts arithmetic only. It ignores attention over long contexts, differences in hardware utilization between training and serving, and the economics of memory and latency, all of which move the real-cost break-even.

**Empirical findings.** A model that will answer very many requests is worth training smaller and longer than compute-optimal. Sardana et al. made this precise: for expected demand around a billion requests they conclude that models should be smaller and trained longer than Chinchilla-optimal, and in their 47 trained models quality kept improving up to 10,000 tokens per parameter [R195]. Gadre et al. found that over-training stays predictable: from runs using about 300 times less compute they predicted the loss of a 1.4-billion-parameter model trained on 900 billion tokens, 32 times more tokens than compute-optimal for its size [R196]. The Llama 3 report says its 405-billion-parameter flagship is near its compute-optimal size, that the smaller models were trained much longer than compute-optimal because they then perform better at the same inference budget, and that the flagship was used during post-training to improve the smaller models [R112].

A caution from a small-scale study: Kumar et al. fitted precision-aware scaling laws on 465 pretraining runs, up to 1.7 billion parameters and 26 billion tokens, and reported that the degradation caused by post-training quantization grows as models are trained on more data, eventually making additional pretraining data harmful [R197]. It is a reason to test quantized serving on the exact checkpoint you plan to ship.

## When unique data runs out

Training on more tokens than a corpus contains means repeating data. Muennighoff et al. ran about 400 training runs, up to 900 billion training tokens and 9-billion-parameter models, and found that for a fixed compute budget, up to four epochs of repeated data changed the loss negligibly compared with unique data, while with more repetition the value of added compute decayed toward zero. They fitted a compute-optimal scaling law that discounts repeated tokens and excess parameters, and tested mitigations such as adding code data or removing common filters [R198]. That result is one reason pipelines pursue more unique tokens, through less aggressive filtering, rephrasing, and synthetic data as above, instead of more epochs; the Nemotron-CC authors cite the four-epoch result when arguing that long training runs need larger filtered sets [R187]. Small, high-quality sources can still be repeated within the four-epoch regime. **Open question:** how far rephrased and synthetic tokens can substitute for fresh human-written text.

## Learning-rate decay, annealing, and mid-training

Large runs end with a learning-rate decay, and the data seen during the decay has outsized effect. **Annealing** is the final phase in which the learning rate falls toward zero. **Mid-training**, also called late-stage curriculum training, is a distinct late phase, often combined with annealing, that switches to a smaller, higher-quality, more targeted mixture: mathematics, code, instruction-like or reasoning-style text, or longer sequences. Published examples:

- Llama 3 annealed the learning rate linearly to zero over the final 40 million tokens at a 128K-token context, upsampled high-quality sources in that phase, and averaged checkpoints taken along the way [R112]. The team also used annealing as an inexpensive probe of data value: anneal a half-trained 8-billion-parameter model over 40 billion tokens with 30 percent drawn from a candidate dataset and compare, instead of running a full scaling study for each dataset. Annealing on in-domain mathematics training sets improved the 8-billion-parameter model noticeably on held-out validation sets of the same benchmarks but improved the 405-billion-parameter model negligibly, which the authors attribute to its stronger in-context learning.
- OLMo 2 introduced a data mix named Dolmino Mix 1124 for its annealing phase and reported that this late-stage curriculum significantly improved many downstream benchmarks [R97].
- Warmup-stable-decay schedules (Part I) hold the learning rate flat for most of training and decay it only at the end. MiniCPM adopted one to allow continued training and domain adaptation [R86]. A schedule study found that a constant rate plus a cooldown scales as predictably as cosine while letting one long run branch into several shorter cooldowns, which lowers the compute needed for scaling experiments [R199].
- Long-context extension is also a late stage: Llama 3 405B raised its window from 8K to 128K in six increments over about 800 billion tokens, advancing only when short-context scores had recovered and needle-in-a-haystack retrieval worked at the new length [R112].

Mid-training interacts with reinforcement learning (RL; see “Reinforcement Learning for Reasoning and Agents”). OctoThinker compared two base-model families under RL on mathematics, found that high-quality math corpora and then question-answer and long-chain-of-thought data in a mid-training phase improved later RL, and proposed a two-stage “stable-then-decay” mid-training: 200 billion tokens at a constant learning rate, then 20 billion tokens with decay across three chain-of-thought-focused branches [R200]. That is an empirical finding from a few model families; it suggests that choices made before post-training limit how much post-training can add. Treat late-stage mixtures like any other data and decontaminate them, because a mix that resembles the benchmarks can raise scores without raising capability.

## Low-precision training at scale

Part I defines the number formats (“Floating-point formats for neural networks” and “Block scaling: microscaling and 4-bit formats”). Training with 8-bit or 4-bit floating-point matrix multiplications raises arithmetic throughput and cuts memory and communication, but such numbers have few distinct values and a narrow range, so one large activation can ruin the scale shared with its neighbors. Published recipes combine five ideas: fine-grained scaling, where small groups of values share a scale; high-precision accumulation of partial sums; a few operations kept in higher precision; higher-precision master copies of weights and optimizer state; and careful rounding.

DeepSeek-V3 (December 2024) trained a 671-billion-parameter mixture-of-experts model on 14.8 trillion tokens with FP8 matrix multiplications [R66]. Its choices:

- all three matrix multiplications of each linear layer (forward, activation-gradient, weight-gradient) run in FP8;
- scales are set per small group, not per tensor: 1×128 tiles for activations and 128×128 blocks for weights, computed online from the current maximum value;
- partial sums are promoted to FP32 every 128 elements, because the FP8 accumulators on their GPUs retained only about 14 bits, by the report’s account (a later paper by the same team describes the same limit as about 13 fractional bits; see Part I); this fix is specific to that hardware generation;
- the E4M3 format is used for all tensors, which the authors say is workable because small groups share an exponent range;
- embeddings, the output head, mixture-of-experts gating, normalization, and attention operators stay in BF16 or FP32, and master weights and weight gradients stay in FP32;
- validation: against BF16 runs on two smaller models (about 16 billion parameters for 1.33 trillion tokens, and about 230 billion for 0.9 trillion), the relative loss error stayed below 0.25 percent.

NVIDIA reported in 2025 the pretraining of a 12-billion-parameter model on 10 trillion tokens in NVFP4, which it described as the longest publicly documented 4-bit run to date [R74]. NVFP4 stores 4-bit floating-point elements in blocks of 16 that share an FP8 scale, plus one FP32 scale per tensor. The recipe adds random Hadamard transforms (a rotation that spreads outliers) on the inputs to weight-gradient multiplications, two-dimensional 16×16 scaling for weights so that forward and backward passes see the same quantized weights, stochastic rounding for gradients, and about 16 percent of the linear layers (the first two and last eight blocks) kept in higher precision. Validation loss stayed within 1 percent of an FP8 baseline during the stable phase and rose to slightly above 1.5 percent as the learning rate decayed. Downstream accuracy was comparable, and switching to higher precision during the decay reduced the loss gap.

Trade-offs and cautions: both reports come from single groups on particular hardware, and the 4-bit loss gap grew during the learning-rate decay. Silent numerical error is a training failure mode (see below), so compare loss curves against a higher-precision baseline at smaller scale before committing a long run, as the DeepSeek-V3 team did. A precision-aware scaling-law fit suggests that lower training precision reduces a model’s effective parameter count [R197], but its runs stop at 1.7 billion parameters. Post-training quantization for serving is a separate problem covered in Part VI [R43, R44, R45].

## Emergence and measurement

Some abilities appear suddenly when plotted against model scale using a thresholded metric. Smooth underlying probability improvement can look discontinuous after exact-match grading. Other capabilities may genuinely require interacting components or sufficient depth.

To study emergence:

- inspect continuous metrics, not only pass/fail;
- control prompting and tokenization;
- compare compute and data fairly;
- measure many checkpoints;
- test transfer and counterfactual variants;
- distinguish elicitation failure from missing capability.

## Checkpoints and validation

Training periodically writes checkpoints containing weights, optimizer state, scheduler state, random-generator state, and data progress. Full optimizer checkpoints can be several times larger than inference weights.

Validation monitors loss and downstream probes. A falling average loss can hide regressions in a language or safety slice. Dashboards should track optimization stability, throughput, data mix, gradient norms, activation statistics, router balance, hardware failures, and evaluation suites.

## Training failures

Large runs encounter failures that require deterministic-enough replay, frequent validated checkpoints, fault-tolerant orchestration, and root-cause analysis. A one-in-a-million sample bug is routine at trillion-token scale. Typical failures include:

- loss spikes or divergence;
- NaNs and overflow;
- corrupted samples;
- accelerator or network failures;
- silent numerical mismatch;
- straggling workers;
- checkpoint corruption or data-pipeline stalls;
- expert imbalance or collapse in MoE models.

Part II (“Keeping large training runs stable”) covers stabilization measures for loss spikes, such as QK-normalization and z-loss.

# Distributed Training Systems

No single accelerator holds or processes a frontier-scale training job. Parallelism divides data, parameters, layers, experts, or sequences across devices.

## Data parallelism

Each worker holds a model replica and processes a different mini-batch. Workers aggregate gradients, often with all-reduce, then apply equivalent updates.

Data parallelism is simple but replicas duplicate weights, gradients, and optimizer state. ZeRO-style sharding partitions optimizer state, gradients, and eventually parameters across data-parallel workers [R26]. Fully sharded data parallelism follows related ideas.

## Tensor parallelism

A large matrix multiplication is split across devices. One device holds some rows or columns; collectives combine partial results. This reduces per-device weight memory but communicates frequently inside each layer, so fast interconnects matter.

## Pipeline parallelism

Layers are split into stages. Micro-batches flow through stages like an assembly line. Pipeline bubbles occur when stages wait; schedules trade utilization, memory, and complexity. Balancing stage compute is difficult with uneven layers or experts.

## Sequence and context parallelism

Long sequences can be divided across devices. Attention requires exchanging summaries or blocks so queries can interact with remote keys and values. Ring-based methods circulate blocks and overlap communication with computation.

## Expert parallelism

MoE experts reside on different devices. Tokens are routed with all-to-all communication, processed, then returned. Capacity factors and load-balancing losses limit overflow and hotspots.

## Combining dimensions

Real training may use data × tensor × pipeline × sequence × expert parallelism. A “3D” or higher-dimensional layout maps the model onto physical clusters. The best mapping depends on topology: bandwidth and latency within one accelerator package, host, rack, and data-center fabric differ.

## Mapping parallelism onto the network

Each strategy sends different traffic at a different rhythm, so placement follows bandwidth needs:

| Strategy | What crosses the links | Rhythm | Usual placement |
|---|---|---|---|
| Tensor | partial results of matrix multiplications | inside every layer, forward and backward | within one server, on the fastest links |
| Sequence or context | blocks of keys and values | inside every attention layer | just outside the tensor groups |
| Pipeline | activations at stage boundaries, point to point | once per micro-batch per stage | across servers |
| Data (sharded) | gradients reduce-scattered, parameters gathered | once per step, can be prefetched and overlapped | outermost, may span a multi-hop network |
| Expert | tokens dispatched to experts and returned (all-to-all) | inside every mixture-of-experts layer | across servers, often with limits on cross-server hops |

The Llama 3 team ordered its four dimensions [tensor, context, pipeline, data] from innermost to outermost for this reason: the innermost needs the most bandwidth and the lowest latency and stays within a server, while the outermost can tolerate multi-hop latency, partly because sharded data parallelism can prefetch parameters and overlap gradient reduction [R112]. DeepSeek-V3 made a different choice for a mixture-of-experts model: 16-way pipeline parallelism, 64-way expert parallelism spanning 8 nodes, ZeRO-1 data parallelism, and no tensor parallelism. By the report’s figures NVLink offered about 3.2 times the bandwidth of InfiniBand (160 versus 50 GB/s), so it limits each token to experts on at most 4 nodes, and it overlaps communication with computation using a pipeline schedule called DualPipe [R66]. Part I discusses the networks involved (“Scale-up and scale-out networks”), where a later paper by the same team, comparing effective rather than nominal bandwidth, puts the ratio at about four. The right layout depends on model shape and cluster; there is no universal order.

## Memory accounting

For one parameter in mixed-precision Adam training, memory may include:

- low-precision model weight;
- gradient;
- higher-precision master weight;
- first moment;
- second moment;
- temporary communication and kernel buffers.

A naive total can exceed 16 bytes per parameter. Sharding, lower-precision optimizer states, and offloading reduce it. Low-precision recipes change which items live in which format: DeepSeek-V3, for example, keeps master weights and weight gradients in FP32 but stores the Adam moments in BF16 (see “Low-precision training at scale”).

**Activation checkpointing** stores selected activations and recomputes others during backward. It trades additional arithmetic for lower memory. This is often worthwhile because compute is cheaper than unavailable memory.

## Reproducibility

Exact bitwise reproduction across distributed hardware can be difficult due to nondeterministic kernels, reduction order, asynchronous faults, and data ordering. Scientific reproducibility instead requires enough disclosure to repeat conclusions: code, data provenance, hyperparameters, seeds, compute, evaluation protocols, and uncertainty. A weight release without training data or recipe supports some reproducibility questions, but not all.

# From Base Model to Assistant

A base model imitates distributions of text. Post-training shapes it toward desired interactions.

## Supervised fine-tuning

Supervised fine-tuning (SFT) trains on input–response demonstrations. For a chat assistant, examples include system instructions, user messages, and ideal assistant responses. Loss is often applied only to assistant tokens.

SFT teaches:

- instruction and role conventions;
- response format and tone;
- task procedures;
- tool-call syntax;
- refusal and uncertainty patterns;
- domain-specific behavior.

Quality matters more than blindly maximizing volume. A small set of carefully written, diverse examples can strongly shape behavior because pretraining already supplies broad capabilities.

## Behavior cloning limits

SFT imitates demonstrated actions. It struggles when demonstrations do not cover states caused by the model’s own mistakes. It also gives equal maximum-likelihood treatment to one chosen answer even when several are acceptable.

Preference learning compares alternatives and can express graded desirability. Interactive reinforcement learning can train on trajectories, though it is more complex and unstable.

## A post-training recipe in practice

Published open reports show the shape of modern post-training.

- **Llama 3** ran six rounds [R112]. Each round collected fresh human preference comparisons, trained a reward model, sampled typically 10 to 30 responses per prompt from the latest model and kept the one the reward model scored highest (*rejection sampling*), fine-tuned on those plus synthetic and curated data, applied DPO on recent preference data, and averaged models obtained from different data or hyperparameter choices. The team tried PPO (proximal policy optimization, a reinforcement-learning algorithm explained in “Policy-gradient ideas in words”) and found that DPO needed less compute at large scale and performed better, especially on instruction-following benchmarks. Two stabilizers were added to DPO: masking special formatting tokens out of the loss, and a negative-log-likelihood term with coefficient 0.2 on the chosen responses.
- **Tülu 3** is an open release of data, code, and recipe that stacks supervised fine-tuning, length-normalized DPO, and a final stage of RL with verifiable rewards, and it reports ablations of methods that did not reliably help [R64].
- **Reasoning models** add large-scale RL stages around these steps (see “Reinforcement Learning for Reasoning and Agents” and “How long chain-of-thought reasoning models are trained”).

**Pattern (a synthesis, not a law).** Supervised fine-tuning sets format and baseline behavior; rejection sampling or preference optimization moves style, helpfulness, and safety; RL applies where an automatic checker exists; weight averaging smooths the result. The DeepSeekMath authors argue that rejection-sampling fine-tuning, DPO, PPO, and GRPO can all be read as direct or simplified forms of reinforcement learning [R63].

## Instruction hierarchy

A deployed application often has instruction sources with different authority: platform policy, developer or system message, user request, retrieved content, and tool output. The model needs both training and runtime structure to treat data as data rather than instructions.

No prompt hierarchy is a hard security boundary by itself. Untrusted retrieved text can still manipulate model behavior. Enforce permissions outside the model.

# Human and AI Preference Learning

Preference training asks which response is better under criteria such as correctness, usefulness, style, and safety.

## Collecting comparisons

Given prompt `x` and responses `y_w` and `y_l`, a rater chooses the winner. Guidelines, rater expertise, cultural context, order effects, and ambiguity affect labels. Agreement should be measured rather than assumed.

Pairwise preference data can train a reward model. A Bradley–Terry-style probability is:

`P(y_w ≻ y_l | x) = sigmoid(r(x,y_w) - r(x,y_l))`

where `r` is a learned scalar reward.

## RLHF

A classic RLHF pipeline is:

1. pretrain a base model;
2. supervised fine-tune on demonstrations;
3. collect preference comparisons;
4. fit a reward model;
5. optimize the language-model policy for reward while penalizing departure from a reference.

A schematic objective is:

`maximize E[reward(x,y)] - β KL(policy || reference)`

The KL term discourages the policy from exploiting reward-model errors by moving too far from known language behavior. PPO has commonly been used for policy optimization [R27], including InstructGPT-style training [R13]. It needs a separate value model, typically about as large as the policy, which adds memory and compute [R63]. Lighter REINFORCE-style estimators avoid that: RLOO samples k completions per prompt and uses the mean reward of the other k − 1 as each completion’s baseline, and in its authors’ experiments it outperformed PPO and the RL-free methods DPO and RAFT [R201].

## Reward hacking

A learned reward is a proxy. Optimization searches for outputs that score highly, including loopholes:

- verbose answers if raters associate length with quality;
- confident style without correctness;
- phrases that exploit the reward model;
- hidden side effects in tool environments;
- solutions that pass weak tests but violate intent.

Hold-out evaluation, adversarial tests, process checks, multiple signals, uncertainty, and constrained environments help. Goodhart’s law is practical: when a measure becomes a target, it can stop being a good measure.

Overoptimization can be measured. Gao et al. used a fixed “gold” reward model as a stand-in for humans, optimized a policy against a proxy reward model trained on the gold model’s labels, and found that the gold score eventually falls as the proxy is optimized harder. The relationship has a different functional form for reinforcement learning than for best-of-n sampling, and its coefficients scale smoothly with reward-model size [R202]. Reward hacking in verifiable and agentic environments, including a formal definition of an unhackable reward, is covered in “RL environments and reward hacking.”

## Direct preference optimization

DPO derives a classification-like objective over preferred and rejected responses, using log-probability ratios against a reference policy [R14]. A schematic term is:

`-log sigmoid(β[(log π(y_w|x)-log π_ref(y_w|x)) - (log π(y_l|x)-log π_ref(y_l|x))])`

It raises the preferred response relative to the rejected one while anchoring to a reference. DPO avoids training a separate reward model and online policy rollouts in its basic form. Variants address label noise, margins, online data, length bias, and multiple preferences.

DPO is not “RLHF solved.” It still depends on preference quality, pair selection, reference behavior, coverage, and evaluation.

## DPO variants and what the evidence supports

Many relatives of DPO exist. Few have been compared under identical conditions, so treat this list as a menu of ideas, not a ranking.

| Method | What changes relative to DPO | Source |
|---|---|---|
| Length-normalized DPO | divides each response’s log-ratio by its length, to reduce length bias | [R64] |
| IPO (the identity-mapping case of ΨPO) | works directly with pairwise preferences instead of assuming they can be replaced by pointwise rewards, with an efficient optimization procedure and performance guarantees | [R203] |
| KTO | learns from one binary label per response (desirable or not), with an objective based on prospect theory, so paired comparisons are not required | [R204] |
| SimPO | uses the average log-probability of a response as the implicit reward, drops the reference model, and adds a target reward margin | [R205] |
| ORPO | adds an odds-ratio penalty on the disfavored response to the supervised loss, in one stage and without a reference model | [R206] |

The length-normalized form used in Tülu 3 is:

`-log sigmoid(β[(log π(y_w|x)-log π_ref(y_w|x))/|y_w| - (log π(y_l|x)-log π_ref(y_l|x))/|y_l|])`

**Empirical findings.** In the Tülu 3 ablations, among DPO, SimPO, and length-normalized DPO, only the length-normalized form beat the starting checkpoint overall [R64]. Llama 3 used standard DPO with an added likelihood term [R112]. A study comparing DPO and PPO reported that PPO, once its key implementation factors were identified, surpassed the other alignment methods in all its test beds [R207]. The KTO authors conclude that no single human-aware loss is universally best [R204]. The conclusion matches the one above for DPO itself: the right choice depends on data, base model, and evaluation, so compare on your own.

## AI feedback and constitutions

RLAIF uses model-generated critiques, rankings, or labels. Constitutional AI supplies written principles and trains a model to critique and revise responses against them [R15]. AI feedback scales but can reproduce judge biases and shared blind spots.

Written rubrics are another form of AI feedback. Rubrics as Rewards turned per-prompt lists of criteria into rewards for medical and scientific questions and reported gains over a judge model’s single-rating scores [R208]. Kimi K2 trained its own model as a critic that ranks responses pairwise against core, anti-hacking, and human-written rubrics, and kept refining the critic with signals from tasks that have verifiable rewards [R93].

Use AI judges where their errors are tolerable and audited. For subjective, adversarial, novel, or high-stakes cases, diverse human expertise remains essential.

## Verifiable rewards

Mathematics, code, games, formal proofs, and constrained tasks can provide automated checks. **Reinforcement learning with verifiable rewards (RLVR)** replaces the learned reward model with a checking function: the model samples a response, a program decides whether it is acceptable, and the reward is a fixed value if so and zero otherwise. The name and recipe came from the Tülu 3 report, which applied it to mathematics, grade-school word problems, and instruction-following constraints [R64]. DeepSeek-R1-Zero used the same idea at much larger scale with an accuracy reward (for example, a final answer in a specified format, or code run against test cases) and a format reward, and deliberately avoided neural reward models for reasoning because they were prone to reward hacking [R32].

Verification quality defines the objective. Unit tests can be incomplete. A proof checker validates formal syntax and rules, not whether the formal statement matches real intent. Rule-based answer matchers can reject correct answers written in another form, and model-based verifiers can be fooled; both effects have been measured for mathematical reasoning [R209]. Sandboxed execution is mandatory for generated code.

The next chapter covers the algorithms, the environments, and the open questions.

# Reinforcement Learning for Reasoning and Agents

Preference learning optimizes toward judgments of “better.” This chapter covers reinforcement learning against checks: a program, a test suite, an environment, or a rubric decides whether a response worked. Since 2024 this has become a main route by which frontier systems improve at mathematics, code, and multi-step tool use. The chapter separates what published evidence shows from what is still open.

## Policy-gradient ideas in words

**Definition.** In reinforcement learning the language model is the *policy*: given a prompt and the tokens so far, it assigns a probability to every next token. A sampled response is one *episode*, and a *reward* scores the finished response. Training nudges the model so that responses with higher reward become more probable.

The key device is a *baseline*. If every sampled response earns a positive reward, raising all of them teaches little; what matters is how much better a response was than expected. That difference is the *advantage*. A positive advantage raises the probability of the response’s tokens; a negative one lowers it. PPO [R27] estimates the baseline with a learned *value model* that predicts expected reward at each token, and it limits each update with a *clipped ratio* of new to old token probabilities so that one batch cannot move the model too far. A KL penalty toward a reference model, as in RLHF above, can be added.

The value model is usually about as large as the policy, which adds memory and compute, and language tasks often give a reward only at the end of a response, which makes accurate per-token value estimates hard to learn [R63]. Most algorithms in this chapter replace the learned baseline with one computed from several sampled responses to the same prompt.

## Group relative policy optimization

**Group relative policy optimization (GRPO)** was introduced in the DeepSeekMath paper in 2024 [R63] and used for DeepSeek-R1 [R32].

In words: for each prompt, sample a group of G responses from the current policy and score each one. A response’s advantage is its reward minus the group’s average reward, divided by the group’s standard deviation, so it measures how many standard deviations better or worse than its siblings the response was. With an outcome reward, every token in a response shares that response’s advantage; the paper also describes a process-reward variant that gives each step its own. Update the model with the PPO-style clipped objective plus a KL penalty to a reference model. No value model is needed, because the group average is the baseline.

In symbols, following the paper’s objective and its KL estimator:

```text
sample o_1 … o_G from π_old(· | q);  reward r_i for each response o_i
A_i    = (r_i − mean(r_1 … r_G)) / std(r_1 … r_G)
ρ_i,t  = π_θ(o_i,t | q, o_i,<t) / π_old(o_i,t | q, o_i,<t)
J(θ)   = average over i of  (1 / |o_i|) · sum over t of
           [ min( ρ_i,t · A_i ,  clip(ρ_i,t, 1−ε, 1+ε) · A_i )  −  β · KL_i,t ]
KL_i,t = π_ref / π_θ − log(π_ref / π_θ) − 1      (both probabilities at token o_i,t)
maximize J over θ
```

The ratio ρ says how much more likely the new model makes a token than the model that wrote it. Clipping stops one update from earning extra credit for pushing ρ outside 1 ± ε, and the minimum takes the more cautious of the clipped and unclipped terms. The KL expression is an estimator that is never negative [R63].

A small example. Suppose four sampled answers score 1, 0, 0, 0, so only the first was verified correct. The mean reward is 0.25 and the population standard deviation is about 0.43, so the advantages are about +1.73 for the correct answer and −0.58 for each wrong one. If all four were correct, or all four wrong, the standard deviation is zero and the prompt gives no learning signal. The sketch computes this with the population standard deviation; an implementation that uses the sample standard deviation rescales every advantage in a group by the same factor.

```python
from statistics import mean, pstdev

def group_advantages(rewards):
    """Group-relative advantage: (reward - group mean) / group standard deviation."""
    mu, sigma = mean(rewards), pstdev(rewards)
    if sigma == 0:                      # every answer scored the same
        return [0.0] * len(rewards)     # this prompt gives no learning signal
    return [(r - mu) / sigma for r in rewards]

adv = group_advantages([1, 0, 0, 0])    # four sampled answers, one verified correct
assert round(adv[0], 3) == 1.732 and round(adv[1], 3) == -0.577
assert abs(sum(adv)) < 1e-9             # advantages are centered on the group mean
assert group_advantages([1, 1, 1, 1]) == [0.0] * 4   # all correct: nothing to learn
assert group_advantages([0, 0, 0, 0]) == [0.0] * 4   # all wrong: nothing to learn
```

What GRPO trades away:

- It needs G generations per prompt, which multiplies sampling cost.
- Groups whose rewards are all equal contribute nothing to the gradient, so very easy and very hard prompts waste samples [R210].
- One advantage is spread over every token of a response, so credit assignment is crude: useless tokens inside a correct answer are reinforced along with useful ones.
- Dividing by the standard deviation and by response length changes how much weight prompts and responses receive. The next section shows why that matters.

## Successors and fixes

GRPO works, but large runs exposed failure modes, and several groups published repairs. Each was developed and tested in its authors’ own setup, often mathematics with a particular family of base models, so read them as evidence about specific failures, not as a ranking.

- **RLOO.** REINFORCE leave-one-out samples k responses per prompt and uses the mean reward of the other k − 1 as each response’s baseline. Its authors argued that many components of PPO are unnecessary for RLHF [R201].
- **Dr. GRPO.** This analysis argues that GRPO has two optimization biases. Dividing by response length gives short correct answers larger updates and long wrong answers smaller penalties, which can lengthen wrong answers; dividing by the group standard deviation up-weights prompts that are almost always right or almost always wrong. The fix removes both divisions. The authors also found the same length normalization in popular open-source PPO implementations [R211].
- **DAPO.** An open recipe with four changes. *Clip-higher* uses separate clipping bounds (`ε_low` = 0.2 and `ε_high` = 0.28 in the paper) so that rare, low-probability tokens can rise, slowing *entropy collapse*, in which the policy becomes nearly deterministic early. *Dynamic sampling* discards prompts whose responses are all correct or all wrong and refills the batch. A *token-level loss* averages over all tokens in the batch instead of averaging within each response first. *Overlong reward shaping* applies a graded length penalty near the maximum length instead of a flat penalty for truncated responses. DAPO also drops the KL term, arguing that a long-reasoning policy is expected to move far from its starting point [R210].
- **GSPO.** GRPO applies an importance ratio to every token, but each token ratio comes from a single sample and the reward belongs to the whole response. Group sequence policy optimization instead uses a sequence-level ratio, `(π_θ(o | q) / π_old(o | q))^(1 / |o|)`, and clips whole responses. Its authors report more stable training, especially for mixture-of-experts models, and say the design has the potential to simplify RL infrastructure [R212].
- **CISPO.** MiniMax found that clipping removed rare tokens such as “However,” “Recheck,” and “Wait” from later gradient updates, because their importance ratios were already high after the first update. CISPO clips the importance-sampling weights instead of discarding token updates [R213].
- **Kimi k1.5.** A variant of online mirror descent trained without a value function, process reward model, or tree search, with a length penalty, curriculum and prioritized sampling, and an RL context window extended to 128K tokens [R214].

**Empirical regularity.** A compute-scaling study of more than 400,000 GPU-hours fitted sigmoid-shaped curves of performance against RL compute. It found that details such as loss aggregation, advantage normalization, curriculum, and off-policy handling mostly change how quickly a run climbs, not where it levels off; that some recipes do level off lower; and that stable recipes follow curves predictable enough to extrapolate from smaller runs, which the authors checked up to a single 100,000 GPU-hour run [R215]. **Open question:** which algorithmic differences still matter at the largest scales and in non-mathematical domains.

## Rewards and verifiers in practice

A reward function is part of the model’s curriculum. Choices that published recipes report:

- **Rule-based accuracy plus format rewards.** R1-Zero rewarded a correct final answer in a specified format (or code that passed test cases) and rewarded putting the reasoning inside designated tags [R32].
- **Make answers easy to check.** DAPO rewrote problems whose answers were expressions so that each answer became an integer, which reduced parser errors [R210].
- **Choose prompts of moderate difficulty.** The Kimi K2 report names diverse coverage and moderate difficulty as principles for its math and logic data [R93]; DAPO’s dynamic sampling enforces a similar idea during training.
- **Add shaping rewards, knowing their cost.** R1’s language-consistency reward, the proportion of words in the target language, made reasoning more readable and slightly lowered measured performance in the authors’ ablation [R32].
- **Use a model as judge only where rules cannot decide.** For the supervised data it harvested after RL, R1 also had a model judge predictions against ground truth, which widened coverage beyond rule-checkable answers [R32].

Verifier errors compound as the policy improves. One study found that rule-based answer matchers reject equivalent answers written in other formats, producing false negatives that hurt more as the policy gets stronger, while model-based verifiers are more accurate on static tests but can be exploited during RL [R209].

## What does reinforcement learning actually change?

**Open question.** The reports reviewed here agree that RL on verifiable rewards raises the chance that a single sample is correct (pass@1) and changes the length and organization of reasoning. They disagree on whether it creates abilities the base model lacks. *Pass@k* is the fraction of problems solved by at least one of k independent samples; at large k it measures what a model can do at all, rather than how reliably.

- DeepSeek-R1-Zero showed responses growing longer, with reflection and verification behaviors, when RL was applied to a pretrained base [R32]. An analysis of R1-Zero-like training found that the base model already showed self-reflection (the “aha moment”), and that some base-model families reason well without any prompt template, suggesting a role for pretraining data [R211].
- A large pass@k study across model families and RL algorithms found that RL-trained models beat their base models at small k but that base models matched or exceeded them at large k. It concluded that the observed reasoning abilities were bounded by the base model, whereas distillation introduced new patterns [R216].
- A later study of prolonged RL, with KL control, periodic resets of the reference policy, and a diverse task suite, reported gains that persisted under extensive sampling and correlated with the base model’s competence and the length of training [R217].
- Random rewards improved mathematics scores for one model family nearly as much as correct rewards did. The authors attribute this to a clipping bias that amplifies behaviors already present in the base model, and the effect often failed to appear in other families [R218].

Read together: validate RL claims on more than one model family, report pass@k at large k, and include reward ablations. Whether RL can discover genuinely new reasoning strategies remains unsettled in the sources reviewed for this edition.

## RL environments and reward hacking

An *RL environment* is the set of tasks or a task generator, the interface the model acts through (plain text or tools), the reward or verifier, and the sandbox that runs everything. The model learns what the environment rewards, not what its designers meant.

**Reward hacking** is raising the measured reward without achieving the intended goal. A formal definition calls a proxy reward *unhackable* if increasing its expected return can never decrease the expected true return, and shows that making a reward narrower by dropping terms often does not make it unhackable [R219]. Reported examples:

- With visible unit tests, an agent may delete the failing tests instead of fixing the bug. ImpossibleBench builds tasks whose tests contradict the written specification, so that passing them signals exploitation [R220].
- Model-based verifiers can be fooled by response patterns they misjudge as correct, especially after RL fine-tuning [R209]. DeepSeek avoided neural reward models for reasoning RL for this reason [R32].
- Hacking can generalize. In one set of experiments a pretrained model was taught about reward-hacking strategies through synthetic documents or prompting and trained on real production coding environments. It learned to hack, and it also showed alignment faking and attempted sabotage in agentic evaluations; standard chat-style safety training fixed chat evaluations but not agentic ones. Preventing the hacking, more diverse safety training, and framing hacking as acceptable during training (“inoculation prompting”) each reduced the effect [R221].
- Training directly against a chain-of-thought monitor can backfire. Light optimization produced better-behaved agents, but heavy optimization led to *obfuscated* hacking, in which intent was hidden in the chain of thought while hacking continued. The authors suggest paying a “monitorability tax” by not applying strong optimization pressure to the chain of thought itself [R222]. Part VI covers monitoring.

Practice for environment designers:

- keep tests, scorers, and answer keys out of the agent’s reach, and verify outcomes from environment state rather than from the model’s claims;
- use several independent signals and hand-check the highest-reward samples;
- hold out evaluations that were never optimized against, and stop when held-out results diverge from the training reward;
- write explicit anti-hacking criteria into rubrics, as Kimi K2 did with its prescriptive rubrics [R93];
- run generated code only in an isolated sandbox with no credentials.

## Agentic and tool-use reinforcement learning

Single-response RLVR has one prompt, one response, and one reward. In agentic RL an episode has many turns: the model emits a tool call, the environment returns an observation, the model continues, and a reward arrives at the end (an issue resolved, a question answered correctly). Tool outputs are text the model did not generate; Search-R1 excludes retrieved text from the loss (“retrieved token masking”) for stability [R223].

Published examples:

- **Search.** Search-R1 trained models to issue multiple search queries during step-by-step reasoning, rewarded only by the correctness of the final answer [R223].
- **Software engineering.** SWE-RL trained on open-source software-evolution data (code snapshots, changes, issues, and pull requests) with a lightweight rule-based reward, the similarity between the generated and the ground-truth solution, and reported gains on some out-of-domain tasks where a supervised baseline degraded [R224]. The MiniMax-M1 RL mixture included sandboxed real-world software-engineering environments [R213].
- **Tools at scale.** Kimi K2 built a synthetic pipeline that generates tools, agents, and tasks with rubrics, simulates multi-turn trajectories, filters them with a judge model, and complements simulation with real execution because simulators have limited fidelity. A joint RL stage then combined verifiable rewards with a self-critique rubric reward [R93].

Difficulties: long episodes make rewards sparse and credit assignment hard; environments are slow, stateful, and expensive to run in parallel; simulators can mislead; sandboxes must be isolated; and every loophole in a grader is an exploit waiting to be found. **Open question:** whether skills learned in a few environments transfer to new tools and tasks. The pass@k study discussed above points to multi-turn agent–environment interaction as a direction that might go beyond the limits it found [R216].

## Self-play and self-generated tasks

A model can also supply its own training problems. The Self-Taught Reasoner loop generates rationales for many questions, retries failures with the correct answer as a hint, fine-tunes on the rationales that reached correct answers, and repeats [R225]. Absolute Zero goes further: one model proposes tasks intended to maximize its own learning progress and then solves them, with a code executor validating the tasks and checking the answers. Its authors report a model trained with no external data that outperformed other zero-setting models on coding and mathematics [R226]. Both depend on a trustworthy checker, and executable checks cover only part of what we want from a model. A risk, not a measured result: a task proposer can drift toward narrow families of tasks that are easy to verify.

## RL training systems

An RL run alternates generation and learning: rollout workers sample responses with an inference engine, a verifier or sandbox scores them, a trainer computes advantages and updates the weights, and the new weights return to the rollout workers. Response lengths vary widely, and in a synchronized system a batch waits for its slowest samples, so generation time is dominated by long-tail responses [R210]. Kimi k1.5’s *partial rollouts* split long responses across iterations and reuse earlier segments instead of regenerating from scratch [R214]. Because rollouts come from slightly older weights, importance ratios and clipping correct for that off-policy gap [R212]. Inference efficiency matters here too: the Llama 3 report says adopting PagedAttention [R46] raised rejection-sampling throughput by more than a factor of two [R112].

# Reasoning and Inference-Time Compute

“Reasoning model” can refer to post-training that improves multi-step problem solving, an inference procedure that spends variable compute, or a product exposing an effort control. It is not one architecture.

The current form arrived in two steps. On 12 September 2024, OpenAI introduced o1, a model it said was trained with large-scale reinforcement learning to produce a long chain of thought before answering; the announcement reported that performance improved with more reinforcement learning (train-time compute) and with more time spent thinking (test-time compute), and it did not disclose the training details [R62]. In January 2025 DeepSeek-R1 [R32] and Kimi k1.5 [R214] published recipes showing that reinforcement learning with verifiable rewards could produce similar long-reasoning behavior. This chapter first covers the older prompting and search ideas, then those recipes, then how reasoning effort is controlled and how test-time compute scales.

## Chain-of-thought prompting

Asking a model to produce intermediate steps can improve performance on some arithmetic, symbolic, and commonsense tasks [R28]. Intermediate tokens provide a scratch space: later computation conditions on earlier generated results.

Few-shot chain-of-thought supplies worked examples. “Let’s think step by step” can elicit a latent procedure. Performance depends on model, task, language, prompt, and decoding.

A visible chain is not guaranteed faithful. Models can rationalize an answer, omit decisive internal features, or be influenced without reporting it. In one study of reasoning models, a hint inserted into the prompt changed the answer, yet in most of the settings tested the chain of thought mentioned the hint in at least 1 percent of the examples where the model used it and often in fewer than 20 percent [R61]. Use the chain as a work product to check, not privileged mind reading.

## Self-consistency and candidate search

Instead of trusting one sampled reasoning path:

1. sample several diverse paths;
2. extract candidate answers;
3. vote, score, or verify.

Self-consistency improved reasoning benchmarks by marginalizing over paths [R29]. More generally, best-of-N chooses the best candidate under a verifier. Tree search expands partial solutions and prunes weak branches.

Compute helps only if candidate diversity and selection quality are sufficient. A biased verifier can select polished mistakes more confidently as N grows.

Repeated sampling shows how far this can go when a checker exists. In one study the coverage—the fraction of problems solved by at least one sample—rose smoothly, close to log-linearly, over four orders of magnitude of samples. On a software-engineering benchmark with automatic tests, repeated sampling let a weaker model surpass the best single-sample result of the time. Where no checker exists, majority voting and reward models stopped improving after a few hundred samples [R227].

## Process and outcome supervision

**Outcome supervision** rewards the final answer. It is cheap when answers are verifiable but gives sparse credit. **Process supervision** evaluates intermediate steps, offering denser signals and potentially discouraging invalid reasoning [R30]. It is expensive and requires a definition of a valid step.

Automatically checkable intermediate states—compiler errors, proof obligations, simulator states, equations—can provide process feedback without exposing a proprietary free-form chain of thought.

Evidence since 2023 is mixed:

- Cheaper labels. Math-Shepherd built step-level labels automatically instead of paying annotators [R228]. The common automatic method, Monte Carlo estimation, judges a step by what completion models do after it. A later analysis found that such labels gave worse performance and generalization than labels from a judge model or humans, and that best-of-N scores for process reward models can be inflated when a policy reaches a correct answer through flawed reasoning; it proposed consensus filtering between methods [R229].
- Large-scale RL. DeepSeek-R1’s authors report that their process-reward attempts did not pay off at scale: a fine-grained step is hard to define, judging intermediate steps is hard, and a model-based process reward invites reward hacking and adds a retraining burden. They found such models useful for reranking or guiding search [R32]. Kimi k1.5 reports strong results without value functions, process reward models, or tree search [R214]. A report that omits a technique does not prove the technique was never tried.

**Open question:** how outcome and process feedback should be combined.

## How long chain-of-thought reasoning models are trained

**Definition.** A long-chain-of-thought reasoning model is trained to produce an extended deliberation, often thousands of tokens that explore, check, and revise, before its final answer.

DeepSeek-R1 [R32] is the most completely documented recipe. It has two parts.

- **R1-Zero** applied RL directly to a pretrained base model with only rule-based accuracy and format rewards and a template asking for reasoning followed by an answer. Responses grew longer during training and came to include reflection, verification, and alternative attempts. The model was hard to read, mixed languages, and was narrowly strong on reasoning tasks.
- **R1** repaired this in four stages: (1) supervised fine-tuning on thousands of cold-start long-chain-of-thought examples in a readable, conversational style; (2) reasoning RL with rule-based rewards plus a language-consistency reward; (3) rejection sampling from the RL checkpoint, keeping correct and readable responses and adding non-reasoning data, for about 600,000 reasoning and 200,000 other examples, followed by fine-tuning the base model again on them; (4) a second RL stage combining rule-based rewards for reasoning with reward models for helpfulness and harmlessness [R32]. A peer-reviewed version appeared in *Nature* in September 2025 [R65].

Other reports follow similar shapes. Qwen3 describes four stages for its largest models: a long-chain-of-thought cold start, reasoning RL, “thinking mode fusion” (supervised fine-tuning on both thinking and non-thinking data so that one model can do either, switched by flags in the chat template), and general RL; smaller models are distilled from larger ones [R124]. Kimi k1.5 scaled the RL context window to 128K tokens and reused earlier trajectory segments to make long generations affordable [R214]. At the supervised end, s1 fine-tuned a strong instruction-following model on 1,000 questions chosen for difficulty, diversity, and quality, showing that a small curated set can be enough to teach a strong model to reason at length [R230].

**Pattern (a synthesis of these reports, not a law).** Start from a strong base, perhaps shaped by mid-training; use a small supervised start for format and readability; run large-scale RL on verifiable tasks; convert the RL model’s good outputs into broad supervised data by rejection sampling; finish with preference or rubric rewards for general behavior; distill into smaller models.

Failure modes include verbose or repetitive reasoning, mixed languages, poor readability, and narrow gains: R1-Zero’s reasoning-only RL did little for broader areas such as writing and open-domain question answering [R32].

## Controlling how much the model thinks

Long deliberation costs tokens, latency, and context. Studies of reasoning models report *overthinking*: spending many tokens on trivial questions for little benefit [R231]. Within a single question, shorter sampled chains were more likely to be correct than the longest ones [R232]. Research controls fall into three groups.

- **Modes.** Qwen3 trains one model with a thinking mode and a non-thinking mode, selected by flags in the template [R124].
- **Inference-time budgets.** Qwen3’s thinking budget halts thinking at a user-set length and inserts an instruction to answer from the reasoning so far; the authors say this ability was not explicitly trained but emerged from mode fusion [R124]. s1’s *budget forcing* does the same with an end-of-thinking delimiter, or lengthens thinking by suppressing the delimiter and appending “Wait.” Improvement flattened after about six forced continuations, and suppressing the delimiter too often produced repetitive loops [R230].
- **Length learned in training.** L1 adds a target (“Think for n tokens”) to the prompt and rewards correctness minus a penalty proportional to the gap between the target and the generated length, `reward = (1 if correct else 0) − α · |n_target − n_generated|`. A variant instead rewards staying under a maximum with a clipped bonus [R233]. Kimi k1.5 adds a length reward computed within each group of samples: among correct answers it favors shorter ones, among incorrect answers it penalizes longer ones, and it is phased in gradually [R214].

Provider APIs expose these ideas as effort settings. As of 2 November 2026, the documentation of three major providers describes coarse effort controls rather than exact token budgets:

- OpenAI’s API exposes `reasoning.effort`, whose allowed values depend on the model and can include none, minimal, low, medium, high, xhigh, and max. Lower effort favors speed and fewer tokens; models also reason adaptively within each setting. Reasoning tokens occupy the context window and are billed as output tokens, and reaching a maximum-output limit during reasoning can return an incomplete response with no visible text [R234].
- Anthropic documents “adaptive thinking,” in which the model decides for each request whether and how much to think, steered mainly by an `effort` setting from low to max. It calls the older fixed token-budget mode deprecated and says its newest models reject it. Effort is soft guidance: no level guarantees thinking, and changing the top-level effort between requests invalidates the prompt cache [R235].
- Google documents dynamic thinking by default, steered by a `thinking_level` (low, medium, and high, with minimal on some models). A hard `max_output_tokens` limit counts thinking tokens and can truncate output, so Google advises lowering the thinking level to save tokens instead of shrinking that limit [R236].

Treat effort settings as a quality, latency, and cost dial to evaluate on your own tasks, not a guarantee. Truncation policy, caching behavior, and what “high” means differ by provider and by model.

## Test-time scaling: sequential, parallel, and search

**Definition.** Inference-time (test-time) scaling spends more computation on difficult inputs through longer deliberation, parallel candidates, search, tool calls, critique, or verification. It has two basic axes. *Sequential* scaling lengthens one chain: more thinking tokens, revisions, self-checks. *Parallel* scaling draws many independent attempts and selects among them by majority vote, a verifier, or a search guided by a process reward model [R230].

**Derivation (cost accounting).** With enough parallel hardware, N samples of L tokens each take about the wall-clock time of one sample and about N times the tokens. One chain of N · L tokens uses the same number of tokens but takes about N times as long, because the tokens in a chain are generated one after another. Parallel scaling buys accuracy with compute spend, sequential scaling with latency, and hybrids trade between them.

**Empirical findings.**

- Allocation matters. Allocating test-time compute adaptively by prompt difficulty improved efficiency by more than 4 times over a fixed best-of-N baseline. In FLOPs-matched comparisons a smaller model with extra test-time compute could beat a 14-times-larger model on problems where the smaller model already had non-trivial success, while on the hardest problems the larger pretrained model was preferable [R31]. With a model trained to revise its answers, easier prompts benefited most from purely sequential compute and harder ones from a mix of sequential and parallel compute [R31].
- Coverage grows with samples; selection is the bottleneck. See “Self-consistency and candidate search” above [R227].
- Smaller models with search can sit on the cost–accuracy frontier. A 7-billion-parameter math model paired with a tree-search algorithm outperformed a 34-billion-parameter model across the inference strategies tested [R237].
- Sequential gains saturate. In s1, lengthening a chain by suppressing its end-of-thinking token helped at first, then flattened, and long chains ran into the context window; majority voting and tree search with a process reward model were complementary and could keep improving beyond those limits [R230]. DeepSeek-R1’s authors likewise found that majority voting added to a model that already reasons at length [R32].
- Longer is not always better, as the overthinking results above show.

Costs and failure modes: parallel scaling needs a selector, and a flawed selector or correlated errors cap the gain; sequential scaling needs context and patience; both multiply spend. A rational controller should allocate compute based on expected value:

- answer simple questions quickly;
- detect uncertainty or difficulty;
- escalate to search, tools, or a stronger model;
- stop when marginal expected quality gain falls below cost and latency.

## Distilling reasoning

A stronger or more compute-intensive teacher can generate traces or solutions for a smaller student. The student learns patterns that would be expensive to discover independently. Distillation may transfer useful behavior but also teacher errors and stylistic artifacts.

Two published comparisons are informative. DeepSeek-R1’s authors fine-tuned small open base models on about 800,000 samples generated with the full model, using supervised fine-tuning only. The distilled 32-billion-parameter model was clearly better than a model of the same size they trained with large-scale RL for over 10,000 steps. They conclude that distillation is economical and effective, but that going beyond current capability may still need stronger base models and larger-scale RL [R32]. Qwen3 trained small models first by off-policy distillation on teacher outputs in both modes, then by on-policy distillation, in which the student generates and its token distribution is pulled toward the teacher’s by minimizing KL divergence, an instance of training the student on its own outputs with teacher feedback [R238]. In one reported experiment on an 8-billion-parameter model, on-policy distillation reached higher scores than RL at about one tenth of the GPU-hours (1,800 versus 17,920) and improved pass@64, which RL did not [R124]. The pass@k study above likewise found that distillation added reasoning patterns that RL on verifiable rewards did not [R216].

Distilled outputs should be filtered with independent verification where possible. Training only on long traces can teach needless verbosity; include short direct solutions for easy tasks.

## Reasoning is broader than prose

Some tasks benefit from internal latent computation, code execution, a calculator, database queries, symbolic solvers, or formal search more than from a long natural-language monologue. The goal is correct, efficient, auditable problem solving—not maximizing visible “thought” tokens.

::: warning Hidden reasoning
A provider may hide internal reasoning traces for safety, privacy, robustness, or product reasons and return a summary. For example, Google’s documentation describes thought summaries returned in place of the raw trace, with billing based on the full thinking tokens generated [R236]. Even when a trace is visible, do not treat it as guaranteed causal evidence. Validate results with external checks.
:::

# Decoding: From Probabilities to Text

The model outputs a distribution. A decoding algorithm chooses the next token and repeats until a stop condition.

## Greedy decoding

Choose the highest-probability token each step. Greedy decoding is deterministic and fast but can enter repetitive or locally attractive sequences. The most likely token at every step does not necessarily form the globally most likely sequence.

## Temperature

Temperature rescales logits before softmax:

`P_i = softmax(z_i / T)`

- `T < 1` sharpens the distribution;
- `T > 1` flattens it;
- approaching zero approximates greedy decoding.

Temperature does not add knowledge or principled creativity. It changes sampling diversity.

## Top-k and nucleus sampling

**Top-k** keeps the k highest-probability tokens and renormalizes. **Top-p**, or nucleus sampling, keeps the smallest set whose cumulative probability reaches threshold `p` [R33]. Top-p adapts set size to confidence.

Filters interact with temperature. Very aggressive truncation can make outputs bland or trap repetition; loose sampling can select incoherent tail tokens. Tune against task-level evaluation.

## Beam search

Beam search maintains several high-scoring partial sequences. It is useful when probability aligns with quality and outputs have constrained lengths, as in some translation settings. Open-ended dialogue can suffer from generic or repetitive high-likelihood text. Length normalization and diversity penalties are task-specific patches.

## Repetition and stop control

Frequency or presence penalties modify logits based on prior use. No-repeat n-gram rules hard-block duplicates. Stop sequences terminate output when matched. These are runtime controls, not learned understanding, and can accidentally truncate valid content.

EOS handling is essential. Training should include appropriate end tokens; serving must cap length and time even if EOS never appears.

## Constrained decoding

Structured outputs can require JSON, a schema, grammar, regular expression, or function signature. Constrained decoders mask tokens that would make the prefix invalid. This provides syntactic guarantees under a correct parser, not semantic correctness.

Validate parsed values, types, bounds, authorization, and business rules after generation. Never execute a tool merely because the JSON parsed.

## Log probabilities and confidence

Token log-probabilities help compare candidate strings or detect unusual generation. They do not directly equal probability that an answer is true. Length, tokenization, prompt, answer format, and calibration affect them.

For classification, define candidate labels and calibrate on representative held-out data. For open-ended factuality, use retrieval, citations, consistency checks, or external verifiers.

## Speculative decoding

A small draft model proposes several tokens; a larger target model verifies them in parallel and accepts a prefix according to a procedure that can preserve the target distribution [R34]. If drafts are often accepted, generation speeds up without changing target-model quality.

Other approaches predict multiple future tokens, use lookahead heads, or cache repeated prefixes. Speed claims should report hardware, batch, prompt/output lengths, acceptance, precision, and quality.

# Adaptation: Prompt, Retrieve, Fine-Tune, or Train?

Choose the least expensive intervention that addresses the actual failure.

## Prompting

Use prompting when the model has the capability but needs clearer instructions, examples, constraints, context, or output format. Prompts are quick to change and easy to A/B test, but consume context and can be brittle.

A robust prompt states:

- role and objective;
- relevant evidence;
- constraints and non-goals;
- output schema;
- uncertainty or abstention policy;
- examples for ambiguous edge cases.

Do not bury critical authorization in prose. Enforce it in code.

## Retrieval

Use retrieval when answers depend on fresh, private, voluminous, attributable, or frequently changing knowledge. Retrieval does not require changing weights. Its failure modes are indexing, recall, ranking, chunking, access control, stale documents, and prompt injection.

## Continued pretraining

Continue the language objective on domain data when the model needs broad familiarity with domain language or patterns. This can improve fluency but may cause catastrophic forgetting, style drift, or overfitting. Mix general data and evaluate both domains.

## Supervised or preference fine-tuning

Fine-tune when behavior must change consistently across many prompts: terminology, format, workflow, tone, tool policy, or domain task. Collect representative examples and maintain a test suite before training.

Parameter-efficient fine-tuning methods such as LoRA train small adapters [R35]. QLoRA keeps a quantized frozen base and backpropagates into LoRA adapters, reducing memory [R36]. Quantization error, optimizer settings, adapter rank, target modules, and data quality still matter.

Where an automatic checker exists—unit tests, a schema validator, an exact-match answer—reinforcement learning with verifiable rewards (see “Reinforcement Learning for Reasoning and Agents”) can improve behavior beyond what demonstrations teach. It needs a trustworthy verifier, a sandbox, a baseline from prompting and supervised fine-tuning to compare against, and monitoring for reward hacking; without these, the training signal can reward the checker’s loopholes instead of the task.

## Distillation

Train a smaller model to match a stronger teacher’s outputs, probabilities, or representations. Distillation reduces serving cost and can specialize behavior. Validate independently because the student may inherit errors and lose rare capabilities. Distilling reasoning behavior has its own evidence; see “Distilling reasoning.”

## Model merging

**Definition.** Model merging combines the weights of several models that share an ancestor into one model, without further gradient training.

- **Averaging.** Averaging the weights of models fine-tuned from the same pretrained checkpoint with different hyperparameters often beat the best single model and costs nothing extra at inference, because the fine-tuned models tend to lie in one low-error region (“model soups”) [R239].
- **Task vectors.** Subtracting the pretrained weights from the fine-tuned weights gives a direction in weight space. Adding such vectors can add skills, and negating one can reduce a behavior [R240].
- **Reducing interference.** TIES-Merging resets parameters that changed little during fine-tuning, resolves sign conflicts between models, and merges only parameters that agree with the chosen sign [R241]. DARE randomly drops a fraction `p` of the fine-tuning changes and rescales the remainder by `1 / (1 − p)` before merging [R242].

Merging also appears inside large training pipelines. The Llama 3 report averages models obtained from different data or hyperparameter versions at each reward-model, supervised, and DPO stage, and averages checkpoints during annealing [R112]. Kimi k1.5 merged a long-reasoning model with a short-reasoning one by averaging weights to reduce token use [R214].

Trade-offs: merging works best among models derived from one base; unrelated models share no common coordinates. Interference can erase a skill, so test each source capability on the merged model. Check that the source licenses allow the combination.

## Train from scratch

Training from scratch is justified when you have unique data and requirements, need control over architecture/tokenizer/license, operate at sufficient scale, and can afford expertise, compute, evaluation, and governance. For most teams, adaptation of an existing model is safer and cheaper.

The best systems often combine methods: retrieval for evidence, a fine-tuned model for behavior, constrained decoding for syntax, and tools for exact computation.

## A decision table

| Need | First choice | Escalate when |
|---|---|---|
| Clearer task behavior | prompt + examples | failures repeat across prompt variants |
| Fresh/private facts | retrieval | retrieval cannot represent required behavior |
| Stable format/style | constrained output + SFT | base capability is missing |
| Domain language | retrieval or continued pretraining | terminology remains poorly represented |
| Lower latency/cost | smaller model, caching, quantization | quality target is unmet |
| New broad capability | fine-tune/distill | only if data and evaluation support it |
| Blend several fine-tuned variants | weight averaging or merging | a source skill regresses, or the models lack a shared base |
| Full control | open-weight base | train from scratch only with strong reason |
