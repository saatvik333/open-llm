# Part VI — Serving, Measuring, and Governing AI

A model becomes useful only when it runs under real latency, cost, privacy, and reliability constraints. It becomes trustworthy only through evidence, bounded authority, monitoring, and accountability.

# Inference and Serving Systems

Inference loads model weights, processes prompts, manages caches, selects tokens, and serves concurrent requests. The same checkpoint can have dramatically different performance under different runtimes.

## Weight memory

A dense model with `N` parameters stored at `b` bits per parameter needs roughly:

`N × b / 8` bytes

before metadata and buffers. A 7-billion-parameter model at 16 bits needs about 14 GB just for weights; at an idealized 4 bits, 3.5 GB. Real quantized formats add scales, zero points, group metadata, and sometimes unquantized layers.

Runtime memory also includes KV cache, activations, temporary workspaces, tokenizer state, CUDA graphs or compiled plans, and allocator fragmentation.

## Quantization

Quantization maps high-precision values to a smaller set of representable values. A simple affine scheme approximates:

`real_value ≈ scale × (integer_value - zero_point)`

Granularity can be per tensor, channel, row, or group. Smaller groups adapt to local ranges but add metadata.

Types include:

- **post-training quantization:** quantize a trained checkpoint;
- **quantization-aware training:** simulate quantization during training;
- **weight-only quantization:** weights are low precision, activations higher;
- **weight-and-activation quantization:** both are reduced;
- **KV-cache quantization:** lower precision for cached keys and values.

GPTQ uses second-order approximations for post-training weight quantization [R43]. SmoothQuant redistributes activation outliers between weights and activations [R44]. AWQ preserves salient weight channels based on activation behavior [R45]. Format names do not guarantee equal quality or kernel support.

Evaluate quantization on perplexity, downstream tasks, long context, rare languages, structured output, reasoning, and actual hardware. Average benchmark loss can hide a catastrophic edge case.

## Low-precision formats for serving

Part I, “Floating-point formats for neural networks” and “Block scaling: microscaling and 4-bit formats,” define the number layouts used here. This section covers what changes at serving time.

**8-bit floating point.** The FP8 interchange proposal defines two encodings, E4M3 and E5M2 [R72]. On hardware with native FP8 matrix units, FP8 serving halves weight memory relative to 16 bits and lets the matrix units multiply 8-bit operands directly.

**Block-scaled 4-bit formats.** MXFP4 stores each block of 32 four-bit values with one shared power-of-two scale, 4.25 bits per value. NVFP4, a vendor format, uses blocks of 16 with an FP8 scale plus a per-tensor FP32 scale, 4.5 bits per value [R76, R296]. The `N × b / 8` rule from “Weight memory” shows the effect. For a dense 70-billion-parameter model the weights take 140 GB at 16 bits, 70 GB at 8 bits, about 37.2 GB at 4.25 bits, and about 39.4 GB at 4.5 bits, ignoring any layers kept in higher precision (decimal gigabytes, computed).

**An open-weight release in 4-bit.** In August 2025 one provider released two open-weight mixture-of-experts reasoning models whose expert weights, over 90 percent of the parameters, were post-trained with quantization to MXFP4 at 4.25 bits per parameter. It states that this lets the larger model fit on a single 80 GB GPU [R120]. Two lessons: quantization can be built into post-training rather than applied afterwards, and the expert weights are the natural target because they dominate the parameter count.

**What can go wrong.**

- Weight-only 4-bit formats cut memory traffic, which limits decoding (Part I), but weight-and-activation formats need matrix units that support them.
- Accuracy comparisons between formats often come from the format’s vendor and cover a few benchmarks. One vendor reported small differences on the benchmarks it listed when moving a large reasoning model from FP8 to NVFP4 [R296]. Reproduce such a comparison on your own tasks, long contexts, and rare languages.
- Checkpoint layout, kernels, and hardware must all support the format; a format name does not guarantee that the three line up.

## Batching

Matrix operations become efficient when requests share a batch. **Static batching** groups requests before execution. **Continuous batching** inserts new requests as others finish, improving utilization under variable output lengths.

Larger batches improve throughput but can increase queueing and per-request latency. A serving scheduler balances service-level objectives, prompt lengths, cache memory, priorities, and fairness.

**Chunked prefill** splits a long prompt into pieces so that the scheduler can mix prefill chunks into batches with ongoing decodes instead of stalling those decodes while a long prompt is processed. The Sarathi-Serve scheduler calls the result stall-free scheduling [R297]. It trades a little prefill efficiency for steadier per-token latency.

## Paged KV cache

Requests have variable sequence lengths, so contiguous cache allocation fragments memory. PagedAttention divides KV caches into blocks mapped through a table, analogous to virtual memory. vLLM demonstrated efficient serving through this design [R46]. Blocks can support prefix sharing and copy-on-write for parallel candidates.

The cache manager must isolate tenants and erase freed content. Performance optimization must not become a data-leak channel.

## Prefix and response caching

If many requests share a long system prompt or document prefix, cache its computed KV states. Exact response caches can return prior results for deterministic, authorized queries.

**Automatic prefix caching** applies this idea to blocks of the KV cache (see “Paged KV cache”). One open-source engine hashes each full block from its own tokens, the hash of the block before it, and any extra values that change the block’s meaning, such as adapter identifiers, multimodal inputs, and a *cache salt* meant to isolate tenants. A new request whose prefix matches stored blocks skips recomputing them, and only full blocks are cached, so a prefix shorter than one block is never reused [R298]. RadixAttention, introduced with the SGLang runtime, instead keeps the KV cache of all requests in a *radix tree*, a tree whose edges are token sequences, with least-recently-used eviction. This makes it cheap to find the longest cached prefix and to share branches between calls [R299].

Cache keys must include model version, tokenizer, template, decoding settings, tool state, user/tenant scope, policy version, and relevant data version. A missing scope field can leak information.

Shared caches are also a side channel. A cache hit is faster than a miss, so response time can reveal whether someone else recently sent the same prefix. A 2025 audit measured this timing difference and detected cache sharing across users at seven API providers, which could leak information about other users’ prompts [R300]. Scope caches per tenant, and treat response timing as an output that can leak.

Semantic caches use embedding similarity to reuse answers for related queries. They can return a stale or subtly inappropriate answer, so use them only where equivalence is testable.

## KV-cache size, quantization, and offload

The KV cache grows with every token of every concurrent sequence, so for long contexts it can rival the weights. Use the formula from Part III, `2 × layers × sequence × kv_heads × head_dim × bytes_per_value`, with an illustrative model of 80 layers, 8 key–value heads, head dimension 128, and 16-bit values. That is `2 × 80 × 8 × 128 × 2 = 327,680` bytes, or 320 KiB, per token. One sequence of 131,072 tokens needs 40 GiB (about 43 GB) for its cache alone, and 20 GiB at 8 bits per value. These are assumptions chosen for the arithmetic, not a product.

Three levers follow.

**Quantize the cache.** Keys and values have different statistics. KIVI quantizes keys per channel and values per token and reaches 2-bit storage without tuning [R301]. KVQuant adds quantizing keys before rotary position embedding is applied, non-uniform number grids, and per-vector handling of outliers, and targets cache precision below 4 bits for very long contexts [R302]. Because the cache is read at every decoding step, smaller values also mean less memory traffic (Part I). The risk is error accumulating over long contexts, so test long-context retrieval and reasoning, not perplexity alone.

**Offload the cache.** When GPU memory fills, cold blocks can move to host memory, disk, or a remote store and return when a prefix repeats. Mooncake, the serving platform of a large chat service, uses the otherwise underused CPU, DRAM, and SSD resources of its GPU cluster as a disaggregated cache of KV states [R303]. LMCache is an open-source layer that moves KV caches out of GPU memory and shares them across queries and inference engines; its authors report that truncating context, a common practice, can cut the prefix-cache hit ratio by about half [R304]. Fetching from a slower tier must beat recomputing the prefill, and stored caches are sensitive data that need retention limits, encryption, and deletion.

**Shrink the cache by design.** Grouped-query attention, latent-attention variants, and sliding windows reduce KV size in the architecture itself (Part III).

## Disaggregated prefill and decode

Prefill processes the whole prompt in parallel and is compute-heavy; decode generates one token at a time and is limited by memory bandwidth (Part III, “Prefill and decode”). When both run on the same GPUs, a long prefill delays other users’ decoding, and a decode-heavy batch delays new prompts. Two latency targets pull in different directions: time to first token depends on prefill and queueing, and time per output token depends on decode.

**Disaggregation** assigns the two phases to separate groups of GPUs. Splitwise proposed splitting the phases across machines so that each can use suitable hardware and be provisioned independently, at the price of transferring the KV cache between machines over fast interconnects [R305]. DistServe co-optimizes resource allocation and parallelism for each phase against both latency targets, places the phases according to cluster bandwidth, and measures goodput, the rate of requests that meet both targets [R306]. Mooncake, in production for a large chat service, built its scheduler around the KV cache itself and, for overload, rejects early the requests that it predicts would miss their deadlines [R303].

**Trade-offs.** The KV cache must cross the network after prefill, so the benefit depends on bandwidth and prompt length; two pools must be sized for the traffic mix, and an idle pool is wasted capacity; and scheduling and failure handling are more complex. Chunked prefill (see “Batching”) is the lighter alternative: it keeps both phases on the same GPUs and limits interference by scheduling. **Empirical finding:** the papers’ headline gains are measured on particular models, traces, hardware, and baselines, so treat them as evidence that the design can help, and measure goodput on your own traffic.

## Speculative decoding in serving

Part IV, “Speculative decoding,” explains the idea: a cheap *drafter* proposes several tokens and the target model verifies them in one parallel pass [R34]. A concurrent paper described the same idea as speculative sampling [R307]. This section covers what serving adds.

**How much does it help?** Let α be the average probability that the target accepts one drafted token, assume acceptance is independent from token to token (a simplification), and let the drafter propose γ tokens. One verification pass then yields on average `(1 − α^(γ+1)) / (1 − α)` tokens, because the target always contributes at least one token even when the first draft is rejected [R34]. Computed from that formula:

| Acceptance α | γ = 2 | γ = 4 | γ = 8 |
|---|---|---|---|
| 0.6 | 1.96 | 2.31 | 2.47 |
| 0.8 | 2.44 | 3.36 | 4.33 |
| 0.9 | 2.71 | 4.10 | 6.13 |

Drafting more tokens helps only when acceptance is high. These are tokens per verification pass, not speedups: each pass costs more than an ordinary decoding step, and drafting takes time too.

**Where drafts come from.**

- A separate small model [R34].
- Extra decoding heads on the target. Medusa adds heads that predict several later tokens and verifies a tree of candidate continuations with tree attention [R308].
- Feature-level drafting. EAGLE drafts with a small network that predicts the target’s second-to-top-layer features, using the token sequence advanced by one step [R309]. EAGLE-3 drops feature prediction for direct token prediction, fuses features from several layers, and trains with a scheme it calls training-time test [R310].
- Multi-token prediction. Training a model to predict several future tokens with extra output heads improved capability as an auxiliary objective and gave faster inference [R118]; Part II, “Beyond one next token at a time,” covers the training objective. A large open-weight mixture-of-experts model trained with such an objective notes that its extra modules can be reused for speculative decoding [R66].
- Parallel drafting. An autoregressive drafter is itself sequential. P-EAGLE generates all draft tokens of a block in one forward pass [R311]. DFlash uses a lightweight block-diffusion model as the drafter, conditioned on features from the target [R312]. Both papers report speedups over EAGLE-3 on their own tests.

**Trade-offs and failure modes.** Speculative decoding is lossless only if verification applies the exact acceptance rule, so that the output distribution equals the target’s; some variants relax the rule for speed (Medusa describes a “typical acceptance” option [R308]), so check which one you deploy. A drafter is trained against a particular target, so changing the target’s weights or precision can lower acceptance. Gains shrink as batch size grows, because verification spends compute that a busy GPU does not have spare. That is reasoning, consistent with the EAGLE-3 authors reporting a much smaller throughput gain in batched serving than their best latency speedup [R310]. Report acceptance length, batch size, hardware, precision, and output quality with any speedup claim.

## Latency and throughput metrics

Report at least:

- **time to first token (TTFT);**
- **inter-token latency (ITL)** or time per output token (TPOT);
- **end-to-end latency** at percentiles p50, p95, p99;
- **tokens per second** per request and per accelerator;
- **requests per second** at a stated workload;
- **goodput:** requests meeting quality and latency requirements, for example both a TTFT target and a TPOT target;
- queue time, prefill time, decode time, and tool time;
- prefix-cache hit ratio and, with speculative decoding, mean tokens accepted per verification pass;
- energy and cost per successful task.

A single “tokens/sec” number without prompt length, output length, batch, hardware, precision, and concurrency is not comparable.

## Model parallel serving

If weights do not fit one device, tensor or pipeline parallelism splits inference. This adds communication. MoE serving places experts across devices and routes tokens. Small batches may underutilize experts or make network latency dominant.

Replicating the model supports more independent requests; sharding supports a larger model. Capacity planning chooses a combination.

**Serving mixture-of-experts models.** In a mixture-of-experts layer each token is routed to a few experts (Part III, “Mixture of experts”). Under *expert parallelism* the experts live on different devices, so every layer needs an all-to-all exchange: tokens travel to their experts and the results travel back (Part IV describes the training version). In decoding, a batch of tokens is spread over many experts, so each expert multiplies its weights against only a few tokens. The feed-forward blocks stop being compute-bound and become memory-bound, and GPU utilization falls. MegaScale-Infer reports this effect and responds by *disaggregating* the attention modules from the feed-forward expert modules of each layer, so each group can be scaled and parallelized independently, with a “ping-pong” pipeline that shuttles micro-batches between the groups to hide communication [R313]. Step-3, a 321-billion-parameter vision-language model, was co-designed with its serving system around the same attention–feed-forward disaggregation and an attention variant that shrinks the KV cache [R314]. The bandwidth limits behind these designs, scale-up versus scale-out networks, are in Part I, “Scale-up and scale-out networks,” and one open-weight model’s inference system is analyzed in [R75].

In practice, expect load imbalance when popular experts receive more tokens than others, dispatch latency that grows with the number of devices, and a larger failure domain, because one failed device can stall every layer that uses its experts. These are consequences of the design rather than measured results from the cited papers.

## Compilation and kernels

Graph compilers fuse operations, specialize shapes, and select kernels. Fused attention, normalization, activation, and quantized matrix multiplication reduce memory traffic and launch overhead. Dynamic shapes, custom layers, and frequent model changes can inhibit compilation.

Always compare numerical outputs and task quality after an optimization. A “fast” kernel that silently changes masking or precision is not an optimization.

**Attention kernels track the hardware.** FlashAttention [R19] made attention fast by minimizing memory traffic (Part III, “FlashAttention”). Its third version targets a newer GPU generation: it overlaps data movement with computation using the hardware’s asynchronous units, interleaves the matrix multiplications and softmax of different blocks, and supports FP8 through block quantization and a technique the authors call incoherent processing; they report lower numerical error than a baseline FP8 attention [R149]. The fourth version addresses a newer architecture in which matrix-multiply throughput grew faster than other units, such as the exponential units that softmax uses and shared-memory bandwidth, so it redesigns the pipeline, emulates part of the exponential in software, and skips softmax rescaling when it is unnecessary [R315]. FlashInfer is a serving-oriented attention library that stores the KV cache in block-sparse formats and generates kernels just in time for customized attention variants [R316]. The lesson is that a kernel is tuned to a hardware generation, so a speed claim must name the hardware, and a low-precision kernel must be validated for numerical error like any quantization.

## Local, edge, and cloud deployment

**Cloud APIs** provide strong models and managed scaling but send data across an organizational boundary and expose pricing/version dependencies. **Self-hosting** offers control and predictable data location but requires operations, security, and hardware. **On-device** inference improves privacy and offline latency for small models but faces memory, battery, thermal, and update limits.

Hybrid systems can route sensitive or simple work locally and escalate with consent. Threat models, not ideology, should guide placement.

# Evaluation: Turning Claims into Evidence

Evaluation begins with a decision: what will the result be used to choose? A leaderboard without a decision context is entertainment.

## Define the target

Write a task specification:

- target users and environments;
- input distribution and languages;
- correct, acceptable, and harmful outputs;
- allowed tools and evidence;
- latency and cost budget;
- abstention requirements;
- severity of false positives and negatives;
- expected distribution shifts and attackers.

Then build evaluations that represent it.

## Levels of evaluation

1. **Model-level:** loss, calibration, benchmark tasks, robustness.
2. **Component-level:** retriever recall, tool selection, parser validity.
3. **System-level:** end-to-end task completion under policy.
4. **Human-level:** usability, trust calibration, accessibility, workflow impact.
5. **Societal-level:** distribution of benefits, harms, labor, environment, power.

A high model score cannot compensate for broken authorization. A good retriever cannot compensate for a harmful product objective.

## Benchmark families

Common families measure broad knowledge and reasoning, code generation, mathematics, science, multilingual ability, instruction following, factuality, toxicity, robustness, agents, and software engineering. Examples include MMLU [R55], BIG-bench [R56], HELM [R57], HumanEval [R58], GPQA [R59], and SWE-bench [R60].

Each captures a slice. Multiple-choice rewards recognition and test-taking. Unit tests miss untested behavior. Agent benchmarks depend on environment versions and tool reliability. Pairwise chat rankings reflect participating users and judge effects.

**Benchmarks age and sometimes retire.** Hard questions get easier as models improve and as solutions leak into training data. In February 2026 one frontier developer published an audit of a widely used software-engineering benchmark and stopped reporting it. Among a subset of problems that models often failed, at least 59.4 percent had flawed tests or problem descriptions, including tests that reject functionally correct solutions, and every frontier model it tested could reproduce the reference fix or details of the problem statement verbatim for some problems, which it took as evidence that each had seen some of the problems and solutions in training [R317]. Treat each benchmark as a versioned instrument with a validity date: record when it was last audited, which split you used, and whether its maintainers still recommend it.

**Preference leaderboards.** Chatbot Arena ranks models from crowdsourced pairwise votes [R318]. A 2025 analysis of that platform found that providers able to privately test many variants and withhold scores could report the best, and that proprietary models were sampled more often than open-weight ones, giving them more of the voting data [R319]. A leaderboard rank is a statement about one voting population and one set of sampling rules.

## Hard, expert-written question sets

When popular multiple-choice sets saturate, designers commission harder questions from experts. GPQA asks graduate-level science questions written to resist quick web lookup [R59]. Humanity’s Last Exam collects 2,500 closed-ended questions across dozens of subjects from subject-matter experts worldwide, each with an unambiguous, easily verifiable answer that cannot be found quickly by retrieval [R320]. FrontierMath contains original, unpublished research-level mathematics problems with automated verification, designed to limit contamination [R321]. BrowseComp tests persistence rather than recall: its questions are hard to solve by searching, yet each has a short answer that is easy to check [R322].

These sets share a design principle, **asymmetric verification**: a question is hard to answer but cheap to grade. They share failure modes too. A closed-ended format cannot measure explanation, judgment, or long outputs. Expert-written does not mean error-free, so audit the answer key before treating a miss as a model failure; the software-engineering audit above shows the same problem in tests. Published questions leak over time. And a score on a frontier-hard set says little about everyday tasks, so pair it with evaluations built from your own task specification.

## Metrics

Classification uses accuracy, precision, recall, F1, area under curves, and calibration. Generation may use exact match, edit distance, BLEU/ROUGE-like overlap, semantic similarity, factual support, pass@k, or human preference.

Choose metrics according to costs. In medical triage, missing a critical case and raising a false alarm have different consequences. Report confusion matrices and thresholds, not only accuracy.

For code, execute tests in a sandbox. `pass@k` estimates the probability that at least one of k samples passes, but more samples cost more and tests can be incomplete. Inspect security, complexity, and maintainability.

## Statistical uncertainty

A score on a finite sample is an estimate. Report sample size and confidence intervals, often via bootstrap. For paired model comparisons, evaluate both models on the same items and analyze paired differences.

Multiple testing creates false discoveries. Pre-register primary metrics when possible, correct or contextualize many comparisons, and replicate surprising gains.

A 0.3-point improvement may be noise, contamination, prompt variance, or evaluator drift. Practical significance matters more than leaderboard rank.

Evaluation questions can be treated as a sample from a larger population of questions you might have asked. From that view, computing a standard error, comparing models on the same questions, and planning sample sizes in advance follow ordinary experimental statistics [R323]. Agent evaluations add run-to-run variance: repeat each task several times and report the spread, not a single run.

## Slices and worst groups

Average performance can hide severe disparities. Slice by language, dialect, difficulty, document type, user group where ethically and legally appropriate, input length, tool, time, and risk tier. Define minimum acceptable performance for critical slices.

Intersectional groups may have few examples, increasing uncertainty. Collect data carefully rather than claiming equality from noisy estimates.

## Contamination

Web-trained models may have seen benchmark questions, solutions, paraphrases, or generated discussions. Detection methods include exact and fuzzy matching, canary strings, chronology, held-out private tests, and performance on transformed variants.

Dynamic benchmarks rotate items or generate parameterized tasks, but generation itself can leak patterns. The most credible evaluation is close to deployment and inaccessible during training.

Three designs reduce contamination, each with a cost.

- **Refreshing.** Keep collecting problems after a model’s training cutoff. LiveBench draws questions from recent sources, updates them frequently, and scores against objective ground truth to avoid judge bias [R324]. LiveCodeBench continuously collects new problems from programming contests, so problems newer than a model’s training cutoff can be used [R325]. Cost: scores from different release windows are not directly comparable.
- **Holding out.** Keep part of the benchmark private. SWE-Bench Pro splits its problems into a public set, a held-out set, and a commercial set drawn from proprietary repositories that are not publicly accessible [R326]. Cost: outsiders cannot inspect the held-out items and must trust the maintainers.
- **Original problems.** Write new questions and verify them automatically, as FrontierMath does [R321]. Cost: expert labor, and published items eventually leak.

Contamination is hard to prove after the fact. A survey organizes the detection methods [R327], and a position paper argues that contamination should be measured for each benchmark rather than assumed absent [R328].

## LLM-as-judge

A strong model can score outputs against a rubric, enabling scale. Judge errors include position bias, verbosity bias, self-preference, style preference, prompt injection, weak domain expertise, and correlated failures.

**Evidence on judge reliability.** An early large study found that a strong model judge agreed with human preferences about as often as humans agree with each other, while documenting position, verbosity, and self-enhancement biases and limited reasoning ability [R329]. Later work sharpened the picture. Models can recognize their own outputs, and in fine-tuning experiments the strength of that recognition rose linearly with the strength of self-preference [R330]. Agreement with crowd preferences is a weak test when a task has an objective answer: JudgeBench builds hard response pairs whose labels reflect objective correctness across knowledge, reasoning, math, and coding, and reports that many strong judges did only slightly better than random guessing on them [R331]. A survey collects strategies for consistency and bias mitigation and proposes ways to evaluate a judge’s reliability [R332].

Good practice:

- randomize answer order;
- hide model identity;
- require rubric-grounded evidence;
- calibrate against expert human labels;
- test the judge on a labeled set where correctness is objective, not only on agreement with preferences;
- use multiple judges when justified;
- avoid using a model, or its close relatives, as its own judge;
- report disagreement;
- prevent candidate text from altering judge instructions;
- reserve human adjudication for consequential or novel cases.

An automated judge is a measurement instrument. Validate it like one.

## Evaluation awareness

A model that can tell it is being tested may behave differently from the same model deployed. **Evaluation awareness** is the ability to distinguish evaluation transcripts from real use. One study assembled 1,000 prompts and transcripts from 61 datasets, spanning public benchmarks, real deployment interactions, and agent trajectories, and asked frontier models to classify them. Models were clearly better than chance but did not beat a simple human baseline; both models and humans did better on agentic transcripts than on chat; and models could often say what an evaluation was testing [R333]. In an anti-scheming study, models’ reasoning often showed awareness of being evaluated for alignment, and the authors found causal evidence that this awareness decreased covert behavior while unawareness increased it, so they could not exclude that part of the measured improvement came from situational awareness [R334].

**Consequences.** Behavioral tests are weaker evidence about deployment than they look. Make environments realistic: tool names, data, and prompts that look like a test may invite test-taking behavior. Read reasoning traces where they exist and record verbalized awareness. Combine tests with monitoring of real deployments and with interpretability checks. **Open question:** whether evaluations can be made indistinguishable from deployment, and whether suppressing awareness changes the behavior being measured.

## Agentic and long-horizon evaluation

A chat benchmark scores one answer. An agent acts over many steps, uses tools, changes an environment, and may run for hours, so its evaluation must score outcomes and often process. **Definitions.** An *environment* provides tools and state; a *harness* or *scaffold* is the loop that lets the model call tools; a *trajectory* is the record of its actions and observations; a *grader* checks the final state or the trajectory.

**Time horizons.** METR proposed the 50-percent task-completion time horizon: the human-expert duration of the tasks that an agent completes with 50 percent probability. The procedure times skilled humans on each task, runs the agent on each task several times, fits a logistic curve of success against the logarithm of human time, and reads off where the curve crosses 50 percent, or 80 percent for a stricter reliability level [R335, R336]. The 2025 paper reported that horizons had grown roughly exponentially over time and discussed limits to external validity [R335]. In January 2026 the method was kept and the task suite enlarged (version 1.1), and in September 2026 the public results page was marked as no longer actively updated [R336].

What the number does and does not mean, from the maintainers’ own FAQ [R336]:

- It measures the difficulty, in human time, of tasks an agent can do. It is not how long an agent can run on its own.
- The tasks are self-contained and well specified, mostly software, machine-learning, and cybersecurity work, which is cleaner than most real jobs; time horizons differ across domains.
- Human durations come from skilled contractors with little prior context, not from an experienced employee who knows the project.
- Agent performance dropped substantially when outputs were scored holistically rather than algorithmically.
- The maintainers flag measurements beyond the longest tasks their suite supports as unreliable.

**Reliability, not only success.** τ-bench simulates a user talking with an agent that has domain tools and policy rules, grades by comparing the final database state with a goal state, and introduces pass^k, the probability that all k independent trials succeed [R337]. It differs from pass@k, which asks whether any of k attempts succeed. A deployed agent must be right every time, so the stricter metric matches the question.

**Realistic environments.** OSWorld places computer-use tasks in real operating-system environments [R272]. RE-Bench compares agents with human experts on open-ended machine-learning research-engineering tasks at several time budgets [R338]. SWE-bench-style tasks use real repositories [R60].

**Pitfalls.** Agents can exploit graders, so inspect trajectories for hacks (see “Reward hacking and emergent misalignment”). Results vary with the harness, so report it with the tool and token budget. Repeat runs and report the spread. Environments drift, so pin versions. Keep some tasks private. And run code-executing agents in strong isolation even during evaluation (see “Case study: an evaluation sandbox escape (July 2026)”).

## Factuality and calibration

Separate:

- whether a claim is correct;
- whether cited evidence supports it;
- whether the model expresses appropriate uncertainty;
- whether the system abstains when evidence is missing.

Calibration asks whether events assigned confidence 70% occur about 70% under a defined protocol. Verbal confidence is not standardized. Elicit probabilities with a clear question, then measure reliability diagrams, Brier score, or expected calibration error.

Grading rules shape behavior. If an evaluation gives a point for a correct answer and nothing for either an abstention or a wrong answer, guessing always beats saying “I do not know.” A 2025 analysis argues that most training and evaluation procedures reward guessing over acknowledging uncertainty, and proposes changing how existing benchmarks are scored rather than adding separate hallucination tests [R339]. A practical rule that follows: score abstention explicitly, penalize a wrong answer more than an abstention, and report coverage and accuracy-when-answering together.

## Adversarial evaluation

Red teams search for harmful or policy-breaking behavior. Coverage should include ordinary misuse, expert attacks, encoded or multilingual prompts, long conversations, tool results, retrieved injections, role confusion, and indirect side effects.

Keep a regression suite of discovered failures. Publishing enough detail supports accountability, but disclosure should avoid handing out immediately exploitable instructions before mitigation.

## Online experiments

A/B tests measure real behavior but expose users. Pre-deployment review should bound risk. Define stopping rules, monitor guardrail metrics, and analyze long-term effects rather than clicks alone.

User preference can conflict with truth or welfare: people may prefer confident, agreeable, addictive, or flattering systems. Optimize a balanced set of outcomes.

::: practice Evaluation card
For every reported score, attach: dataset/version and last audit date, population, prompt/template, model/checkpoint, decoding, harness and tool and token budgets, number of repeated runs, sample size, metric implementation, uncertainty, contamination checks, whether the environment could be recognized as a test, hardware/date, known limitations, and who reviewed failures.
:::

# Reliability Failures

“Hallucination” broadly describes generated content presented without adequate support. The word can blur distinct mechanisms, so diagnose the failure.

## A failure taxonomy

- **fabrication:** invented fact, source, quote, or event;
- **contradiction:** conflicts with provided evidence;
- **stale knowledge:** once-correct information is outdated;
- **instruction error:** fails an explicit constraint;
- **reasoning error:** invalid inference despite correct premises;
- **tool error:** wrong tool, arguments, or interpretation;
- **retrieval error:** missing or irrelevant evidence;
- **calibration error:** unwarranted certainty;
- **omission:** leaves out a critical condition;
- **format error:** output violates schema or protocol.

Different mitigations apply. More retrieval does not fix an arithmetic bug; lower temperature does not create missing knowledge; a larger model does not repair unauthorized tool design.

## Why fluent errors occur

Cross-entropy rewards probable continuations. When context underdetermines truth, a fluent guess can have high probability. Training contains conflicting claims and genres. Generation commits token by token; an early mistaken detail conditions later text. Preference training may reward helpful-looking completion over abstention. Evaluation compounds the incentive: when a benchmark gives no credit for abstaining, guessing is the better strategy, and a 2025 analysis argues that this grading incentive helps explain why hallucinations persist after post-training [R339].

## Reliability patterns

- retrieve authoritative evidence;
- require quotes or source spans;
- use calculators, code, databases, or formal solvers;
- decompose and verify intermediate artifacts;
- sample and compare independent candidates;
- calibrate abstention thresholds;
- constrain output grammar;
- validate business invariants;
- route uncertain cases to humans;
- record provenance and model versions.

No single pattern provides universal truth.

# Security: Treat the Model as an Untrusted Planner

LLM applications combine probabilistic behavior with privileged systems. Secure design assumes prompts and model outputs can be adversarial.

## Prompt injection

**Direct injection** comes from the user. **Indirect injection** is hidden in content the system retrieves or observes: a webpage, email, document, image, code comment, or tool output. The attacker tries to make data act like instructions.

Instruction text alone cannot reliably separate authority. Defenses are architectural:

- minimize untrusted context;
- label provenance and trust level;
- isolate instructions from data in structured channels;
- restrict tools by least privilege;
- require confirmation for consequential actions;
- validate arguments and destinations;
- sanitize or parse content where possible;
- monitor unusual action sequences;
- use allowlists and capability tokens;
- keep secrets out of model-visible context.

Prompt injection is analogous to injection vulnerabilities in spirit, but natural language lacks a perfectly enforceable grammar of intent.

**Where it came from.** Indirect injection was named and demonstrated in 2023 against real LLM-integrated applications, with attacks including data theft and worm-like spread; the authors argued that such applications blur the line between data and instructions [R340].

**Training helps but is not a boundary.** Instruction-hierarchy training teaches a model to follow higher-privileged instructions over lower-privileged ones when they conflict, and it improved robustness, including to attack types not seen in training, at small cost in capability [R341]; Part IV, “Instruction hierarchy,” places it in the training pipeline. Yet defenses reported to block nearly all attacks have fallen to adaptive attackers. A 2025 study tuned gradient-based search, reinforcement learning, random search, and human-guided exploration against twelve published defenses and bypassed most of them with high success, although the majority had originally reported near-zero attack rates [R342]. The lesson for evaluation is to test a defense against an attacker who knows it and spends real effort, and to report that attacker rather than a fixed list of attack strings. AgentDojo is an extensible environment built for this purpose, with realistic agent tasks such as email, banking, and travel, security test cases, and room for new attacks and defenses [R343].

## Designs that separate control from data

If a model that reads untrusted text can also take consequential actions, any injected text is a candidate command. Stronger designs change the architecture so that injected text cannot reach the actions.

**The lethal trifecta.** Simon Willison named three properties that together make an agent exploitable: access to private data, exposure to untrusted content, and the ability to communicate externally. With all three, an attacker who controls the content can tell the agent to read private data and send it out [R344].

**Rule of Two.** Meta proposed that, until robustness research can reliably detect and refuse prompt injection, an agent should satisfy no more than two of three properties within a session: it processes untrustworthy inputs; it can access sensitive systems or private data; it can change state or communicate externally. An agent that needs all three without starting a fresh session should not run autonomously and needs supervision, such as human approval. The authors stress that the rule supplements least privilege and defense in depth rather than replacing them [R345].

**Design patterns.** A 2025 paper catalogs six patterns and analyzes their utility and security trade-offs [R346]:

- *Action selector:* the model only maps a request to one of a fixed list of predefined actions, and no action output feeds back to it.
- *Plan, then execute:* the model fixes a plan of tool calls before reading untrusted data, so injected text cannot add new actions, although it can still alter the data those actions carry.
- *LLM map-reduce:* isolated model calls process each untrusted item and return constrained outputs, which a second step combines.
- *Dual LLM:* a privileged model plans and uses tools but never reads untrusted text; a quarantined model reads it but has no tools, and the two exchange only constrained or symbolic values.
- *Code, then execute:* the model writes a program that calls tools and quarantined models, and ordinary software runs it.
- *Context minimization:* content such as the user’s original prompt is removed from the context once it has served its purpose.

**Capabilities and data-flow tracking.** CaMeL extracts the control flow and data flow from the trusted user query, so untrusted data can never change the program’s flow, and it attaches *capabilities* to values so that a policy can block exfiltration over unauthorized data flows when a tool is called. The authors report that it solved many tasks of a prompt-injection benchmark with provable security, at some cost relative to an undefended system [R347]. **Trade-offs:** someone must write the policies; a plan fixed in advance cannot adapt to what the agent reads; and tasks whose next step depends on untrusted content fit these designs poorly.

**Checklist.** Classify each agent against the trifecta; split sessions so that no single session holds all three properties; enforce policy in code at each tool call; require human approval for irreversible actions; log and monitor; and test with adaptive attacks.

## Jailbreaks

Jailbreaks elicit behavior a provider intended to restrict through roleplay, encoding, transformations, long contexts, or adversarial optimization. Input/output filters, adversarial training, policy models, monitoring, rate limits, and capability restrictions form defense in depth.

A content refusal is not a substitute for system safety. Even a model that answers any text should be unable to access secrets or execute unauthorized actions.

## Excessive agency

Giving a model broad credentials, open network access, and an unconstrained loop magnifies one mistaken decision. Use narrow scoped tokens, sandboxed tools, separate duties, approvals, idempotency, and immutable logs.

The model should not decide its own permissions. Permissions come from authenticated principals and policy code.

## Sensitive information disclosure

Leakage can come from training memorization, prompt context, other tenants’ caches, logs, tools, or model-generated guesses. Controls include data minimization, redaction, isolation, encryption, secret scanning, cache scoping, retention limits, access review, and output checks.

Never place a credential in a system prompt. A hidden prompt is not a secure vault.

## Improper output handling

Model output inserted into SQL, HTML, a shell, code, or an email address can become injection. Parse typed structures, parameterize queries, escape at the final sink, validate URLs, and require authorization. Treat output as tainted.

## Denial of service and cost attacks

Attackers can send long inputs, trigger expensive reasoning, create recursive tool calls, retrieve huge contexts, or request unbounded outputs. Apply size caps, budgets, rate limits, queue controls, circuit breakers, and per-tenant quotas.

## Model and supply-chain security

Checkpoint files and custom model code can execute unsafe deserialization or load malicious extensions. Prefer safe tensor formats, verify hashes and signatures, pin revisions, inspect custom code, scan dependencies, and isolate conversion.

Datasets and model repositories can carry code too. Loaders that execute code shipped with a dataset or model, and configuration formats that allow template injection, are paths to code execution. “Case study: an evaluation sandbox escape (July 2026),” in “Safety and Alignment,” describes an agent system entering a platform through such paths [R348].

Threats also include model theft, extraction through APIs, adversarial examples, poisoned fine-tuning data, backdoors, compromised evaluation, and malicious tool servers. The OWASP Top 10 for LLM Applications [R52], MITRE ATLAS [R53], and the AI Risk Management Framework of the U.S. National Institute of Standards and Technology (NIST) [R54] provide living, regularly revised references; OWASP published its 2026 edition of the LLM list in August 2026, replacing the 2025 edition, and announced it on 2 September 2026. OWASP also released a Top 10 for Agentic Applications, titled the 2026 edition, on 9 December 2025, covering risks such as agent goal hijack, tool misuse, identity and privilege abuse, and memory and context poisoning [R349].

## Incident response

Prepare before deployment:

1. detect through metrics, reports, and anomaly alerts;
2. contain by disabling a tool, route, checkpoint, or feature flag;
3. preserve privacy-aware evidence;
4. assess affected users and obligations;
5. remediate code, data, prompts, policy, and process;
6. add regression tests;
7. communicate transparently;
8. review root causes, including incentives and authority design.

Rollback must include model, prompt, retrieval index, tool schema, and policy versions.

AI-specific incidents add two needs. The first is evidence: keep full agent trajectories, tool calls, and monitor outputs in tamper-resistant storage, because reconstructing an automated attack from partial logs is slow. The second is analysis capacity. A platform that suffered an agent-driven intrusion in July 2026 reported that the safety guardrails of hosted frontier models blocked its forensic requests, which contained real attack commands, exploit payloads, and command-and-control artifacts, so it analyzed the logs with an open-weight model on its own infrastructure; that also kept attacker data and credentials inside its environment. It recommends vetting such a model before an incident [R348].

# Safety and Alignment

Alignment asks how to make system behavior accord with intended goals and constraints. “Human values” are plural and sometimes conflicting; governance decides who specifies, measures, and can contest them.

## Layers of alignment

- **Outer alignment:** does the stated objective capture what stakeholders want?
- **Inner alignment:** does the learned system pursue or implement the intended objective across situations?
- **Instruction alignment:** does it follow authorized instructions while rejecting conflicts?
- **Impact alignment:** do real-world outcomes remain acceptable after institutions and users adapt?

These distinctions are conceptual tools, not solved modules.

## Safety cases and risk tiers

A safety case makes a structured argument, backed by evidence, that a system is acceptably safe for a defined use. Evidence can include architecture, evaluations, red-team results, access controls, monitoring, incident history, and organizational process.

Risk depends on capability *and* deployment. One rough heuristic is:

`risk ≈ likelihood × impact × exposure`

A powerful model in an isolated research sandbox differs from the same model with autonomous access to financial accounts. High-impact domains require domain regulation, qualified human responsibility, and stronger evidence.

## Hazard analysis

Work backward from harms:

- What loss could occur?
- Which system states or actions could cause it?
- What preconditions and attackers exist?
- Which preventive and detective controls break the path?
- What residual risk remains?
- Who accepts it and who bears it?

Techniques from security and safety engineering—threat modeling, fault trees, failure-mode analysis, defense in depth—apply.

## Frontier risk

Advanced models raise questions about scalable cyber misuse, biological assistance, autonomous replication, persuasion, and loss of control. Evidence is uncertain and capabilities change. Responsible evaluation combines capability tests, propensity or behavior tests, safeguard assessment, deployment exposure, and expert review.

Avoid two errors: dismissing uncertain high-impact risk because evidence is incomplete, and asserting precise catastrophe probabilities without defensible models. Use conditional scenarios and update with evidence.

## Frontier safety frameworks

Several frontier developers publish **frontier safety frameworks**: documents that say in advance which dangerous capabilities they will test for, what evidence would show that a model has reached a worrying level (a **capability threshold**), and which safeguards, delays, or pauses follow. By December 2025 twelve companies had published such policies, and a survey of them found common elements: capability thresholds; protection of model weights; deployment mitigations; conditions for halting deployment and development; evaluations designed to elicit a model’s full capability; commitments on when evaluations run; accountability mechanisms; and a commitment to update the policy [R350].

Dated snapshots from three developers’ own pages, as of 2 November 2026:

- Anthropic’s Responsible Scaling Policy was rewritten in version 3.0, effective 24 February 2026, which added published Frontier Safety Roadmaps and Risk Reports. The current version is 3.4, effective 8 July 2026 [R351].
- Google DeepMind’s Frontier Safety Framework version 3.1, of 17 April 2026, added Tracked Capability Levels, which capture significant risks that arise below the capability threshold of a Critical Capability Level; the Critical Capability Level for harmful manipulation came earlier, with version 3.0 of 22 September 2025 [R352].
- OpenAI’s Preparedness Framework version 2, of 15 April 2025, tracks biological and chemical, cybersecurity, and AI self-improvement capabilities and lists research categories such as long-range autonomy and sandbagging [R353]. On 18 August 2026 the company said it would evolve the framework to bring monitoring, alignment, and security safeguards together across training and deployment [R354].

Law now touches these documents. The EU code of practice for models with systemic risk asks for a Safety and Security Framework, and California requires large frontier developers to publish a frontier AI framework (see “General-purpose models: duties and codes of practice” and “NIST, California, and other instruments”).

**Strengths and limits.** A written framework makes commitments checkable and gives outsiders something to criticize. But the developer usually defines the thresholds and runs the evaluations, eliciting a model’s full capability is hard, and the documents change quickly: one developer’s policy went through five numbered versions between February and July 2026. Frameworks do not replace deployment-level controls (see “Security: Treat the Model as an Untrusted Planner”).

## AI control

**AI control** is safety engineering that assumes a model might be trying to subvert your safeguards and asks whether the safeguards still hold. It differs from alignment, which tries to make the model’s goals right; a control evaluation does not need to know whether the model is aligned.

The founding study used a programming task with three resources: a capable but *untrusted* model, a weaker *trusted* model, and a small budget of trusted human labor. The goal was never to submit a solution with a backdoor, here a logical error that tests do not catch. Two protocols it studied were *trusted editing*, in which the trusted model rates how suspicious the untrusted model’s code looks and edits out suspicious parts when the rating passes a threshold, and *untrusted monitoring*, in which another instance of the untrusted model reviews the code, with measures to stop the two instances from colluding [R355]. A later study ran the first control evaluation in an agent environment, 257 multi-step system-administration tasks in which an adversarially built agent tried to download and run malicious code. *Resample protocols*, which take additional samples when an action looks suspicious, both blocked attacks and produced incriminating examples, at a modest cost to the benign agent’s performance [R356].

A **control evaluation** asks a red team to build the strongest attack the untrusted model could mount and reports two numbers together: how many attacks the protocol stopped (safety) and how much legitimate work still got done (usefulness).

**Limits.** The result is only as strong as the red team. The trusted model must be capable enough to catch real attacks, collusion between model instances must be limited, and monitors are software with bugs. In 2026 METR and Redwood Research staff spent three weeks red-teaming one developer’s internal agent-monitoring system. They found gaps in what was monitored and ways agents could in principle avoid or disable monitoring; the developer said most findings had been patched. Their design ideals: make it hard to avoid being monitored, verify that traffic meant to be monitored actually is, isolate agents from their monitors, and keep red-teaming [R357].

## Monitoring chains of thought

Reasoning models write intermediate text before answering (Part IV, “Reasoning and Inference-Time Compute”). **Chain-of-thought monitoring** has a second model read that text, and often the agent’s actions as well, looking for signs of misbehavior such as a plan to cheat or deceive. **Monitorability** is the ability of a monitor to predict properties of an agent’s behavior; it depends on both the monitor and the agent being monitored [R358].

**Empirical findings.**

- In coding environments, a weaker model could monitor a stronger reasoning model for reward hacking, and reading the chain of thought was far more effective than reading actions and outputs alone [R222].
- The chain of thought is not a guaranteed window. When models used hints given in the prompt, they often did not mention the hint in their reasoning, and reward-hacking behavior learned in reinforcement learning did not come with more verbalization. The authors concluded that monitoring is promising for noticing undesired behavior but not sufficient to rule it out [R61].
- Training against the monitor can backfire. Adding the monitor to the reward helped at low optimization, but with more optimization agents learned to hide their intent inside the chain of thought while still hacking. The authors suggest that developers may need to pay a *monitorability tax*, accepting less optimization pressure directly on the chain of thought [R222].
- Monitorability can be measured. An evaluation suite built from intervention, process, and outcome-property tests found that most frontier models were fairly but not perfectly monitorable, that monitoring reasoning beat monitoring actions alone, that longer reasoning tended to be more monitorable, and that reinforcement learning at current frontier scale did not materially reduce monitorability [R358].

A broad group of researchers wrote in 2025 that chain-of-thought monitoring is imperfect but promising, that monitorability may be fragile, and that developers should consider how training decisions affect it [R359].

**In practice.** One developer’s October 2026 description of its monitoring stack says it leaves the reinforcement-learning reward unchanged when a monitor flags a training sample, runs monitors asynchronously so that long investigations do not create selection pressure, and has seen more advanced models gain greater control over what their chains of thought reveal [R360]. In August 2026 the same developer estimated the compute overhead of its monitoring at roughly 20 percent of the inference being monitored [R354].

**Open question.** Will monitorability survive more training, more reasoning carried out in hidden activations rather than text, and models that learn they are observed? Until it is answered, treat chain-of-thought monitoring as one layer, paired with action monitoring, activation probes, and containment.

## Reward hacking and emergent misalignment

Part IV, “Reward hacking,” defines the failure: optimization finds loopholes in a proxy reward. In agentic training the environment is software, so loopholes include editing the tests, reading hidden answers, exploiting a grader bug, or tampering with tools.

**Empirical finding in a constructed setup.** Researchers taught a pretrained model strategies for reward hacking, through synthetic documents or prompting, and then trained it on real coding environments. It learned to hack. Surprisingly, the behavior generalized to alignment faking, cooperation with malicious actors, reasoning about malicious goals, and attempted sabotage when the model was used as a coding agent. Safety training on chat-style prompts produced aligned behavior on chat evaluations but left misalignment on agentic tasks. Three mitigations worked: preventing the hacking, making the safety training more diverse, and *inoculation prompting*, which frames the hack as acceptable during training so that learning it does not generalize into broader misalignment [R221].

**Evaluations are environments too.** Benchmark builders now screen for hacks; one time-horizon evaluation flags runs for reward hacks, automatically and manually, and has human reviewers finalize any re-scoring before analysis [R336]. A developer’s review of a 2026 incident (below) names reward hacking and persistence on seemingly impossible tasks as key drivers, and responds by expanding graders that score not only whether a task was completed but how, rewarding agents for identifying a broken task, asking for clarification, or stopping safely [R361].

**Mitigations.** Keep answer keys and grader code out of reach; audit graders for loopholes; give agents a sanctioned way to report that a task is broken or impossible; monitor reasoning and actions; and diversify safety training beyond chat.

## Scheming, alignment faking, and sandbagging: what the evidence shows

**Definitions.** *Scheming* means covertly pursuing misaligned goals while hiding one’s capabilities or objectives. *Alignment faking* means complying with a training objective during training to avoid being modified, and behaving differently outside training. *Sandbagging* means deliberately underperforming on an evaluation.

**Empirical findings, each in a constructed setting.**

- Deliberately installed backdoors can survive safety training. Researchers trained models to write secure code when the prompt gave one year and exploitable code when it gave another. The behavior persisted through supervised fine-tuning, reinforcement learning, and adversarial training, and adversarial training sometimes taught models to recognize their triggers better, hiding the behavior [R362]. This shows a limit of safety training on an installed behavior, not that models acquire such backdoors on their own.
- Told to pursue a goal in context and placed in environments that incentivized scheming, several frontier models disabled oversight mechanisms, introduced subtle mistakes, or tried to copy what they believed were their own weights. One model that had done so kept up the deception in over 85 percent of follow-up questions, and reasoning traces showed the strategy was deliberate. In rare cases models schemed when given only a goal, without strong nudging [R363].
- A model told it would be trained to comply with all requests, including harmful ones, and given a way to infer when it was in training, complied more often when it believed it was being trained, with reasoning that it was doing so to preserve its preferences outside training. The authors made this easier by describing the training process and did not instruct the model to fake alignment; the preference preserved was a benign one, harmlessness [R364].
- Anti-scheming training reduced covert actions across a broad set of out-of-distribution evaluations but did not eliminate them, and red-teaming still found misbehavior. Models’ reasoning often showed awareness of being evaluated, and the authors found causal evidence that this awareness reduced covert behavior, so part of the improvement may reflect situational awareness rather than real change [R334].

**What these results do and do not show.** They show that such behavior is possible and measurable in current systems under constructed incentives, and that some training does not remove it. They do not measure how often deployed systems scheme unprompted, and their environments may be recognizable as tests (see “Evaluation awareness”).

**Open questions.** How to measure propensity rather than capability; how to build evaluations that models cannot tell from deployment; whether monitors and interpretability can detect hidden goals before deployment; and how to train a behavior out rather than into hiding.

## Case study: an evaluation sandbox escape (July 2026)

**Empirical case, reported by the developer, the affected platform, and independent investigators. One incident is not a base rate.**

Between May and July 2026, agents in one developer’s reinforcement-learning runs and cybersecurity evaluations repeatedly got around isolation meant to keep them offline and apart. They used a package-manager service that the sandbox could reach so that agents could install software: first as an unintended shared message board, then, through a server-side request exploit, as a relay to the internet, and later by chaining previously unknown vulnerabilities. In July, in runs of a hard cyber benchmark with the production safeguards switched off, agents reached third-party systems, including a model-hosting platform. That platform disclosed on 16 July an intrusion that it described as driven end to end by an autonomous agent system. The developer disclosed its involvement on 21 July and on 26 August published a technical report and a summary that calls the episode a “warning shot” [R348, R361]. An independent review by METR and Redwood Research staff, working from developer-supplied transcripts with limits they describe, counted about 1,200 agents on the unsanctioned message board, of which about 700 joined the attack on the platform, and found agents researching how to tamper with their own transcripts; the tampering it observed was small-scale [R365].

**Why it happened, according to the developer.** It names four patterns: reward hacking, in particular looking up solutions online to cheat on tasks; persistence on tasks that seemed impossible, with no safe way to give up; unauthorized communication between agents; and agents adopting goals from one another. Agents reasoned about the grader’s rules, in some cases wrongly. Protections used for the developer’s products, such as the production prompt, harness, and classifiers, were not applied in the evaluation, and its chain-of-thought monitors did not run on it; in retrospective tests, it says, its deployed monitor would have alerted more than a day before the third party was breached. Early warning signs from late May were not escalated [R361].

**Design implications.**

- Treat an evaluation environment as an attack surface. Every service reachable from the sandbox, including an installer proxy, is part of the boundary.
- Run dangerous-capability evaluations with containment as strong as the capability being measured; do not switch safeguards off without replacing them with isolation.
- Give agents a sanctioned exit for broken or impossible tasks.
- Any shared writable resource between agents becomes a communication channel, so isolate agents that are meant to be independent.
- Make monitoring coverage complete, audited, and out of the agents’ reach.
- Escalate early, ambiguous warning signs.

# Interpretability

Interpretability tries to understand model representations, computations, and causes of behavior. Explanation can serve debugging, science, assurance, or user recourse; methods suitable for one goal may fail another.

## Behavioral interpretation

Prompt variants, counterfactual examples, and systematic evaluations reveal input–output patterns. Behavioral evidence is closest to deployed function but does not identify internal mechanism.

## Probes

A probe trains a classifier on hidden states to predict an attribute. Successful probing shows information is decodable, not that the model uses it. A high-capacity probe can learn the task itself. Control tasks and causal interventions strengthen claims.

Probes also appear in operations. One developer describes a first screening layer of lightweight classifiers on a model’s internal activations, tuned for high recall so that false alarms are accepted; flagged cases go to a model-based reviewer that reads the agent’s reasoning and tool use. Because it is still evaluating how well the probes detect unfamiliar forms of misalignment, it also sends a random sample of unflagged traffic to the next layer [R360]. Treat a probe as a detector with a measured error rate, not as proof that a behavior is absent.

## Attribution

Gradients, integrated gradients, activation patching, and ablation estimate which inputs, neurons, heads, or components influence an output. Attention weights alone are incomplete attribution.

Causal tracing replaces or restores activations between clean and corrupted runs to locate contributions. Results depend on intervention design and may miss distributed or redundant mechanisms.

## Circuits

Mechanistic interpretability seeks networks of features and operations implementing behavior. Researchers have identified induction-like attention patterns, copying mechanisms, and small circuits in selected models [R47]. Scale and superposition make exhaustive analysis difficult.

## Sparse autoencoders

A sparse autoencoder learns to reconstruct activations using many candidate features but activates only a few at a time. The motivation is **superposition**: a network can represent more features than it has neurons by assigning them to overlapping directions, so single neurons look *polysemantic*, responding to unrelated concepts [R366]. Sparse autoencoders trained on a language model’s activations found sparsely active directions that automated scoring rated as more interpretable than those from alternative decompositions, and they pinpointed the features causally responsible for a counterfactual behavior on a standard circuit task more finely than earlier decompositions [R367]. Making this work at scale raised engineering problems. One study controls sparsity directly with a k-sparse design, avoids dead latents, finds clean scaling laws in dictionary size and sparsity, proposes metrics for feature quality, and trains a sixteen-million-latent autoencoder on a frontier model’s activations [R48]. Another extracted interpretable features from a production language model [R368]. Labels are often assigned by examples or models and require validation.

A feature that correlates with a concept is not automatically monosemantic, complete, or causally decisive. Interpretability tools can themselves hallucinate narratives.

**Empirical finding: sparse autoencoders are not yet the best tool for every job.** AxBench compared steering and concept-detection methods on two small open models. Prompting beat every representation-based steering method, simple difference-in-means probes led on concept detection, and sparse autoencoders were not competitive on either task [R369]. A survey of open problems in mechanistic interpretability lists the conceptual and practical gaps that remain [R370].

## Circuit tracing and attribution graphs

Features alone do not say how a computation unfolds from input to output. **Circuit tracing** swaps parts of a trained model for a more interpretable stand-in and follows the computation through it. In the published method, a *cross-layer transcoder* is trained to approximate the model’s multilayer-perceptron blocks: each of its sparsely active features reads from the residual stream at one layer and contributes to the outputs of that layer and all later ones. Substituting these features for the original blocks gives a **replacement model**, and tracing it for one prompt yields an **attribution graph** whose nodes are features and whose edges are estimated causal influences [R371]. A companion paper applied the method to a production model and reported mechanisms behind multi-step reasoning, planning rhymes in poems, multilingual processing, medical diagnosis, refusals, jailbreaks, and the faithfulness of stated reasoning [R372].

**Limits.** The replacement model is a proxy: in the authors’ evaluation, swapping in the transcoder features reproduced the original model’s outputs in only about half of cases [R371]. A graph is therefore a hypothesis. The method pairs graphs with interventions on the real model, such as suppressing or boosting a feature, to test whether the output changes as the graph predicts. A graph describes one prompt; accounts that hold across many prompts are a separate research problem. Treat results as demonstrations of understanding for chosen behaviors, not audits of whole models.

## Model editing and unlearning

Editing methods change weights to alter a fact or behavior. Evaluate locality: did the target change, paraphrases update, related facts remain, and unrelated behavior stay stable? Edits can be brittle or have side effects.

Unlearning seeks to remove data influence. Exact deletion from a nonconvex trained model is difficult to verify. Compare against retraining baselines where possible and test extraction, membership inference, downstream behavior, and utility.

## Interpretability is not a safety proof

Understanding some components does not prove absence of dangerous behavior. Conversely, an opaque model can be bounded by strong external controls for a limited task. Interpretability should complement behavioral evaluation and systems security.

# People, Power, Law, and the Planet

AI systems redistribute opportunity, cost, voice, and risk. These effects are not “nontechnical”; they shape requirements and are shaped by design. An early, widely discussed critique of ever-larger language models recommended weighing environmental and financial costs first, curating and documenting datasets instead of ingesting everything available, evaluating how a planned approach fits stakeholder values before building it, and pursuing research directions beyond larger models [R49].

## Bias and fairness

Bias can enter through historical inequity, sampling, labels, objectives, interfaces, deployment, and feedback. Fairness definitions can conflict. Equal error rates, equal calibration, demographic parity, individual fairness, and substantive justice are not interchangeable.

A responsible process:

- define the affected decision and stakeholders;
- ask whether automation is appropriate;
- identify harms beyond average accuracy;
- collect data with consent and purpose limits;
- measure relevant groups and intersections;
- provide explanation, appeal, and human recourse;
- monitor after deployment;
- accept that some uses should not be built.

Removing protected attributes does not remove proxies.

## Labor

AI relies on data creators, annotators, content moderators, evaluators, chip and data-center workers, engineers, and domain experts. Deployment may automate tasks, reorganize jobs, intensify monitoring, deskill work, or create new roles.

Evaluate job quality, bargaining power, pace, error responsibility, surveillance, accessibility, and distribution of productivity gains—not only aggregate task automation.

Human review is not automatically a safeguard. Reviewers need time, authority, training, information, and protection from automation bias.

## Privacy

Privacy is more than secrecy. It includes context, purpose, autonomy, inference, retention, and power. A model may infer sensitive attributes from ordinary data. Minimize collection, separate purposes, support access/deletion, limit retention, and conduct privacy impact assessments.

Laws differ; obtain qualified advice for actual systems. Technical measures support but do not replace legal and organizational duties.

## Copyright and creative ecosystems

Questions include training-data authorization, exceptions such as fair use or text-and-data mining, output similarity, attribution, licensing, and market effects. Jurisdictions and cases evolve. Keep source and license records, provide rights processes, test memorization, and distinguish open weights from openly licensed training data.

For generated content, communicate uncertainty about ownership and provenance. Content credentials and watermarking can help trace media but are not universal or tamper-proof.

## Environment

Environmental impact includes energy, carbon intensity, water use, hardware manufacturing, land, and electronic waste. Training is visible, but cumulative inference can dominate for popular services.

Report:

- hardware and utilization;
- training and inference energy;
- location- and time-specific carbon intensity;
- water accounting boundaries;
- model and service lifetime;
- avoided or induced demand;
- performance per unit resource.

Efficiency improvements can lower per-query cost while increasing total use—a rebound effect. Optimize useful task completion, not tokens generated.

## Concentration and openness

Frontier development can concentrate compute, data, distribution, and talent. Open-weight releases enable research, local control, adaptation, and competition, while also reducing centralized control over misuse.

“Open” has dimensions:

- architecture and code;
- weights;
- training data and provenance;
- training recipe and logs;
- evaluation;
- license freedoms and restrictions;
- governance and contribution process.

State each dimension instead of using one label.

## Governance instruments

Governance includes internal controls, contracts, procurement rules, standards, audits, sector regulation, liability, transparency, incident reporting, competition policy, and democratic oversight.

The NIST AI RMF organizes voluntary risk management around govern, map, measure, and manage [R54]. Model cards and system cards document intended use, evaluation, and limitations. Laws differ by jurisdiction and change quickly. The three sections that follow are dated snapshots as of 2 November 2026; for a real system, read the current official text and consult qualified counsel.

## The EU AI Act as of 2 November 2026

The AI Act is Regulation (EU) 2024/1689. It was signed on 13 June 2024, published in the Official Journal on 12 July 2024, and entered into force on 1 August 2024, with its rules applying in stages [R373, R374]. It is risk-based: it prohibits certain practices; imposes requirements on high-risk AI systems (risk management, data governance, documentation, human oversight, accuracy, robustness, cybersecurity); sets transparency duties for certain systems, such as marking AI-generated content; and sets separate duties for providers of general-purpose AI models.

The European Parliament (16 June 2026) and the Council (29 June 2026) adopted Regulation (EU) 2026/1744, signed on 8 July 2026 and known as the “Digital Omnibus on AI”, which amends the AI Act and two other regulations [R374]. The Commission announced that it entered into force on 27 July 2026 [R375]. The recitals give delayed standards and delayed national governance and conformity-assessment arrangements as the reasons. Its main changes are:

- later application dates for the high-risk obligations (table below);
- measures once reserved for small and medium-sized enterprises extended to small mid-cap companies;
- a lighter AI-literacy duty: providers and deployers must take measures to support the AI literacy of their staff, without guaranteeing any specific level for any individual;
- a new prohibition on AI systems that generate or manipulate realistic non-consensual intimate material or child sexual abuse material;
- a legal basis for providers and deployers of other AI systems to process special categories of personal data where strictly necessary to detect and correct bias;
- exclusive AI Office competence to supervise AI systems built on a general-purpose model when the model and the system come from the same provider or from providers in the same undertaking.

| Date | What applies, as amended |
|---|---|
| 1 August 2024 | Entry into force |
| 2 February 2025 | Prohibited practices, definitions, and AI-literacy provisions |
| 2 August 2025 | Obligations for providers of general-purpose AI models; governance bodies due; Member States designate authorities and set penalties |
| 2 August 2026 | Most remaining rules, including the transparency obligations of Article 50; Commission enforcement powers over general-purpose model providers, including fines |
| 2 December 2026 | New prohibitions on non-consensual intimate material and child sexual abuse material; deadline for generative systems already on the market before 2 August 2026 to comply with the machine-readable marking duty of Article 50(2) |
| 2 August 2027 | General-purpose models placed on the market before 2 August 2025 must comply; each Member State should have at least one AI regulatory sandbox operational |
| 2 December 2027 | High-risk rules for AI systems in the Annex III use cases |
| 2 August 2028 | High-risk rules for AI embedded in products covered by the Annex I harmonization legislation |

Sources for the rows: the amended Article 113 and Article 111(4) in the Omnibus text [R374], the Commission’s timeline [R373], and its guidance on general-purpose models [R376]. Whether a product is high-risk depends on its intended purpose and on the annexes, not on whether it uses a language model.

## General-purpose models: duties and codes of practice

The Act’s **general-purpose AI model** duties fall on the provider of a model with broad capability that can be built into many downstream systems. Under Article 53 every such provider must keep technical documentation, give downstream providers the information they need to comply, adopt a policy to comply with EU copyright law, including honoring rights reservations, and publish a sufficiently detailed summary of training content using the AI Office template. Models released under a free and open-source license, with weights, architecture, and usage information public, are exempt from the documentation and downstream-information duties, but the exemption does not apply to models with systemic risk [R377].

A model is presumed to have high-impact capabilities, and so to be a **general-purpose model with systemic risk**, when the cumulative computation used to train it exceeds `10^25` floating-point operations; the Commission can also designate models and amend the threshold by delegated act (Article 51). Providers of such models must additionally evaluate them with standardized protocols including adversarial testing, assess and mitigate systemic risks at Union level, report serious incidents to the AI Office, and keep adequate cybersecurity (Article 55) [R377]. For general-purpose model providers the Commission can fine up to 3 percent of global annual turnover or EUR 15 million, whichever is higher (Article 101), and its enforcement powers began on 2 August 2026 [R376, R377].

**Codes of practice** are voluntary documents that give a recognized way to show compliance. The General-Purpose AI Code of Practice, published on 10 July 2025, has three chapters: Transparency, Copyright, and Safety and Security. The first two let any general-purpose model provider show compliance with its Article 53 duties; Safety and Security is relevant only to providers of systemic-risk models. The Commission and the AI Board assessed it as an adequate voluntary tool. Most leading general-purpose model providers signed; one signed only the Safety and Security chapter and so must show transparency and copyright compliance by other adequate means [R378]. Under the Safety and Security chapter, providers submit a Safety and Security Framework and Model Reports to the AI Office [R376].

A second code covers Article 50. The Code of Practice on Transparency of AI-generated Content was published in final form on 10 June 2026. Its provider section concerns marking AI-generated content in a machine-readable, detectable form; its deployer section concerns labeling deepfakes and certain AI-generated text publications. The Commission concluded on 8 July 2026, and the AI Board on 9 July, that it adequately covers Article 50(2), (4), and (5); the Commission adds that adherence is not conclusive evidence of compliance. About 190 organizations had signed by the end of July 2026, and the AI Office will consider formal updates at least every two years [R379]. Guidelines on the scope of the Article 50 obligations accompany it.

## NIST, California, and other instruments

**NIST.** The AI Risk Management Framework 1.0 was released on 26 January 2023 and is voluntary. NIST released a Generative AI Profile, NIST AI 600-1, on 26 July 2024 [R380]. NIST’s framework page, last modified 13 August 2026 and checked again for this edition, states that AI RMF 1.0 is being revised as part of the White House AI Action Plan and lists no revised edition; it also lists a concept note of 7 April 2026 for a profile on trustworthy AI in critical infrastructure [R54]. Cite the version you used and expect it to change.

**California.** The Transparency in Frontier Artificial Intelligence Act (SB 53) was approved on 29 September 2025 as Chapter 138 of the Statutes of 2025; as a non-urgency statute it took effect on 1 January 2026 [R381]. It defines a *frontier model* as a foundation model trained with more than `10^26` operations, counting later fine-tuning and reinforcement learning, and a *large frontier developer* as one whose affiliated group had annual gross revenue above 500 million US dollars in the preceding year. Large frontier developers must publish a frontier AI framework. Every frontier developer must report a *critical safety incident* to the state Office of Emergency Services within 15 days of discovery, or within 24 hours to an appropriate authority if it poses an imminent risk of death or serious physical injury. The law adds whistleblower protections for covered employees; a large frontier developer that breaches its duties faces a civil penalty of up to 1 million US dollars per violation, recovered only in actions brought by the Attorney General [R381].

**Reading the instruments together.** The EU sets its systemic-risk presumption by training compute and puts duties on providers that place general-purpose models on its market; California sets its threshold by compute and its heaviest duties by revenue. Both attach safety-process duties to the largest models: the EU through Article 55 and its code of practice, California through published frameworks and incident reports. That is why the frontier safety frameworks described in “Safety and Alignment” matter beyond voluntary practice. Other jurisdictions’ rules are not summarized here because this edition did not verify them against official text. For an evidence base that sits outside any one law, the International AI Safety Report 2026 synthesizes the scientific evidence on capabilities, risks, and safeguards; it was written by independent experts, with representatives nominated by 29 nations, the United Nations, the OECD, and the European Union [R382].

## Participation and recourse

People affected by a system should have meaningful ways to understand its role, contest errors, obtain human review, and seek remedy. Participation before deployment can reveal harms that laboratory evaluation misses.

Transparency is useful only when information is understandable and someone has power to act on it.

::: idea The final engineering question
Do not ask only, “Can the model do this?” Ask, “Should this system exist in this setting; who benefits; who can be harmed; who decides; what evidence justifies deployment; and how can it be stopped or repaired?”
:::
