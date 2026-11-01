# Part VIII — The Frontier and Beyond

The frontier is not one model or one benchmark. It is a moving Pareto surface across capability, efficiency, modality, autonomy, openness, and safety. This chapter separates a dated product snapshot from durable research directions.

# The 2 November 2026 Landscape

Everything in this snapshot was read on official provider documentation, model cards, technical reports, or standards pages on 2 November 2026. Where a page shows its own update date or a changelog gives a release date, the date is stated. Product names are **dated examples**, not rankings, endorsements, or a complete market map. The book gives no benchmark scores, leaderboard positions, or prices: they change faster than the structure they describe, and a score means something only under the exact protocol that produced it (Part VI).

How quickly do names expire? The newest flagship-tier releases named below carry release dates between 1 September and 26 October 2026, so a snapshot taken about ten weeks earlier would have named different current flagship models at OpenAI, Anthropic, and Google. Treat every name as a coordinate in time. The durable content is the pattern.

## Hosted frontier families

- **OpenAI.** As of 2 November 2026 the [model catalog](https://developers.openai.com/api/docs/models) recommends GPT-6 Astra (released 3 September) for the most demanding reasoning and coding work, GPT-6.1 Sol for near-flagship results at lower cost, GPT-6.1 Terra (26 October) for visual reasoning and precise computer use, and GPT-6 Luna for focused, high-volume work. The [flagship model page](https://developers.openai.com/api/docs/models/gpt-6-astra) lists a 1,050,000-token context window (922,000 input plus 128,000 output), text and image input, text output, and reasoning effort from low to max. Separate catalog entries cover live and realtime voice, transcription, image generation, embeddings, moderation, two open-weight models, a cybersecurity line, and a life-sciences model, several with restricted access (see the specialists section below).
- **Anthropic.** The [models overview](https://platform.claude.com/docs/en/models/overview) lists four current tiers: Fable 5.1 (1 September) for demanding reasoning and long-horizon agentic work; Opus 5.5 (22 September), the suggested starting point for most workloads; Sonnet 5.5 (28 September), described as the best combination of speed and intelligence; and Haiku 5.5 (7 October) for high-volume, latency-sensitive work such as classification, extraction, and routing. Each lists a 1M-token context window, 128K output tokens, and adaptive thinking steered by an effort setting. Mythos 5.1 is the same model as Fable 5.1, offered only to organizations verified through the provider’s verification programs.
- **Google.** The [Gemini API model page](https://ai.google.dev/gemini-api/docs/models) (last updated 28 October 2026) describes Gemini 4 Argon as the flagship reasoning model, generally available since 13 October, with a unified 1,048,576-token context window, text, image, video, audio, and PDF input, and text output. Gemini 3.8 Flash is the fast tier aimed at agents and long-horizon software work. The same catalog lists live voice, text-to-speech, transcription, image, video, and music generation, multimodal embeddings, an embodied-reasoning model for robotics, a deep-research agent, and a managed agent that runs in an isolated sandbox.
- **DeepSeek.** The [API documentation](https://api-docs.deepseek.com/quick_start/pricing) exposes two model names, `deepseek-v4-pro` and `deepseek-flash`; the second is currently served by DeepSeek-V4.1-Flash-1017, released on 17 October 2026. Both list a 1M-token context window and up to 384K output tokens, with thinking on by default and an effort parameter. The API accepts OpenAI-style and Anthropic-style request formats, and the weights of both models are published under the MIT license.
- **Mistral.** The [model catalog](https://docs.mistral.ai/models) lists Mistral Large 4 (multimodal; catalog version 26.10; public preview on the API since 6 October, open weights released on 29 October), Mistral Small 4, three Ministral 3 sizes, document OCR, speech transcription and synthesis, moderation, and time-series forecasting models. It also serves third-party open-weight models and labels them as third-party.

Other important commercial and research providers, national labs, universities, open communities, and device makers are not exhausted here, and each catalog omits older models that remain available. The point is the pattern: a provider no longer offers simply “the model.” It offers a **portfolio** organized by capability, latency, modality, safety profile, access conditions, deployment location, and price, and choosing among its tiers is part of application design (Part V, model routing).

## Open-weight ecosystems

An **open-weight** model publishes its trained parameters for download. That is not the same as open source: the training data, code, and recipe may be withheld, and the license may restrict use. On 2 November 2026 the [Hugging Face model hub](https://huggingface.co/models) listed about 3.2 million public models; its live counter read 3,205,089. The hub holds language models, diffusion models, adapters, quantized files, and specialists.

The table lists large open-weight language models whose own model cards or technical reports were read for this snapshot. Active fraction is the active parameters divided by the total, computed here. All are mixture-of-experts models (Part III).

| Maintainer and model | Total parameters | Active per token | Active fraction | Context (tokens) | License |
|---|---|---|---|---|---|
| Moonshot AI, Kimi K3 | 2.8T | 104B | 3.7% | 1,048,576 | custom |
| Alibaba, Qwen3.8-2.4T-A95B | 2.4T | 95B | 4.0% | 262,144 native, extensible to 1,010,000 | custom |
| DeepSeek, V4-Pro | 1.6T | 49B | 3.1% | 1M | MIT |
| Mistral AI, Large 4 | about 1T | 49B (52B with embeddings) | 4.9% | 1M | custom |
| Meta, Muse Spark Open | about 816B | about 42B | 5.1% | up to 1M | custom, gated download |
| NVIDIA, Nemotron 3 Ultra | 550B | 55B | 10.0% | up to 1M | OpenMDW-1.1 |
| MiniMax, M3 | about 428B | about 23B | 5.4% | 1M | custom |
| OpenAI, gpt-oss-120b (August 2025) | 117B | 5.1B | 4.4% | 131,072 | Apache-2.0 |

Sources: the [Kimi K3 card](https://huggingface.co/moonshotai/Kimi-K3), the [Qwen3.8 card](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B), the [DeepSeek-V4 report](https://arxiv.org/abs/2606.19348) [R139] and [V4-Pro card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro), the [Mistral Large 4 card](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B), Hugging Face’s [launch post](https://huggingface.co/blog/meta-muse-spark-open) for Meta’s release (15 October 2026), the [Nemotron 3 Ultra card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4), the [MiniMax M3 card](https://huggingface.co/MiniMaxAI/MiniMax-M3), and the [gpt-oss model card](https://arxiv.org/abs/2508.10925) [R120]. Smaller open-weight families exist at every scale. For example, Gemma 4 ships in five sizes from the E2B size up to 31 billion parameters, including one mixture-of-experts variant (26B A4B), with context up to 256K tokens under Apache-2.0 ([card](https://huggingface.co/google/gemma-4-31B-it)), and Ministral 3 offers 3B, 8B, and 14B models under Apache-2.0.

Three regularities appear in these cards and reports. They are observations about these releases, not laws.

**Total size and active size have separated.** Among the 2026 releases in the table, total parameters range from about 430 billion to 2.8 trillion, and only about 3 to 10 percent are active for each token; the 2025 gpt-oss-120b follows the same pattern at 117 billion total. Memory follows the total; per-token compute follows the active part. A rough estimate shows the consequence. Kimi K3’s card lists 4-bit weights, so its weights alone occupy about 2.8 × 10¹² × 4 ÷ 8 = 1.4 × 10¹² bytes, or 1.4 terabytes, before block scales, the key–value cache, and activations (the formula of Appendix A, and Part VI). Cheap per-token compute does not mean cheap to host.

**Attention is being re-engineered for million-token contexts.** Most of the largest releases replace some or most full-attention layers with cheaper mechanisms. The Kimi K3 card lists 69 linear-attention layers and 24 gated latent-attention layers [R130]; the Qwen3.8 card lists three gated linear-attention layers for every full gated-attention layer; the DeepSeek-V4 report describes compressed sparse attention combined with heavily compressed attention [R139]; the MiniMax M3 card describes a sparse-attention mechanism; and Nemotron 3 Ultra mixes state-space (Mamba-2), attention, and expert layers. The pressure is the quadratic cost of full attention (Part III); the open question is which approximations preserve long-range recall.

**Licenses differ, and the difference matters.** DeepSeek-V4-Pro is MIT-licensed, gpt-oss-120b is Apache-2.0, and Nemotron 3 Ultra’s card names OpenMDW-1.1; the other entries use custom licenses. Texts read for this snapshot include a requirement to display the model’s name on products (for Kimi K3 and Qwen3.8 above 100 million monthly users or 20 million US dollars in monthly revenue; for MiniMax M3 on any commercial use), a separate agreement for companies that sell hosted access to the model (for Kimi K3 and Qwen3.8 above stated revenue thresholds; Mistral’s card states the requirement without a threshold), and a grant limited to non-commercial use unless extra conditions are met (MiniMax M3). A release can be open-weight without disclosing data or permitting every use. Read the exact license, acceptable-use terms, patents, redistribution rules, and derivative obligations, and check whether the tokenizer, inference code, and evaluation recipe are included.

Licenses also interact with law. The EU AI Act’s obligations for providers of general-purpose AI models have applied since 2 August 2025, and the Commission’s enforcement powers over them apply from 2 August 2026 [R376]. Article 53(2) relaxes two documentation duties for models released under a free and open-source license that allows access, use, modification, and distribution, with public weights, architecture, and usage information, but the relaxation does not apply to general-purpose models with systemic risk [R377]. Whether a custom license with user-scale or revenue conditions counts as free and open-source under that text is a legal question to put to counsel, not a question a table can answer.

A few releases publish more than weights. Ai2 states that its Olmo 3 release includes “the entire model flow,” meaning every stage, checkpoint, data point, and dependency used to build the models [R386]. That is a different kind of openness from a weights-only release, because it lets others audit and reproduce the training rather than only run the result.

At the other end of the scale, open-weight models sized for laptops and phones are routine. Runtimes document the paths: [llama.cpp](https://github.com/ggml-org/llama.cpp) with the [GGUF file format](https://huggingface.co/docs/hub/gguf), [MLX LM](https://github.com/ml-explore/mlx-lm) on Apple silicon, [ExecuTorch](https://docs.pytorch.org/executorch/stable/llm/getting-started.html) for exporting to mobile devices, and [LiteRT-LM](https://developers.google.com/edge/litert-lm/overview) for Android, iOS, web, desktop, and embedded devices. Frontier training, however, remains concentrated in a few organizations: the Mistral Large 4 card says that trillion-parameter model was trained from scratch on 3,800 GPUs in the developer’s own datacenters, a cluster size few institutions operate (Part IV).

## Reasoning effort and adaptive thinking

Every provider read for this snapshot exposes a control for **how much the model thinks before it answers**. The names differ; the idea is the one in Part IV: spend more inference computation on hard problems and less on easy ones [R31].

| Provider documentation | Control | Levels documented | Notes |
|---|---|---|---|
| OpenAI | `reasoning.effort`, plus a separate `reasoning.mode` of standard or pro | model-dependent subset of none, minimal, low, medium, high, xhigh, max | flagship accepts low through max; raw reasoning is not exposed |
| Anthropic | `output_config.effort` with adaptive thinking | low, medium, high, xhigh, and max; xhigh is not available on every model | all five levels on the four current tiers; effort scales every output token, including tool calls |
| Google | `thinking_level` | low, medium, high; minimal on some models | dynamic thinking by default; the token-budget parameter `thinkingBudget` is deprecated on the newest flagship |
| DeepSeek | `thinking` enabled or disabled, plus `reasoning_effort` | low, high, max | thinking on by default at high effort; reasoning text is returned |

Two shifts since the first reasoning models stand out. First, controls moved from **explicit budgets** to **coarse levels plus adaptation**. Early interfaces asked the developer to set a number of thinking tokens. Anthropic’s documentation now says its manual thinking-budget mode is not accepted on its latest models, and “adaptive thinking” lets the model decide how much to think, steered by effort. Google’s documentation says its models think dynamically by default and marks the token-budget parameter deprecated for the newest flagship. OpenAI’s documentation says its models “reason adaptively across reasoning efforts.” The trade-off is predictability: a level bounds behavior less tightly than a budget, so applications still need output caps, timeouts, and per-task monitoring of token use.

Second, providers differ in **what the reasoning looks like to the caller**. OpenAI does not expose raw reasoning tokens, offers optional summaries, and returns encrypted reasoning items that stateless applications pass back. Anthropic returns either a summary or an empty field, with an encrypted signature that must be passed back unchanged. Google returns opt-in thought summaries plus encrypted signatures of its reasoning state, and an application that manages conversation state itself must resend the thought blocks exactly as received. DeepSeek returns the reasoning text and requires it to be passed back when tools are in use. At OpenAI, Anthropic, and Google, reasoning the caller cannot read is still billed.

::: warning A reasoning summary is not a trace
A visible summary is generated text. When researchers planted hints in prompts, the reasoning of the models they tested often did not mention the hint even when it changed the answer, with reveal rates often below 20 percent [R61]. A multi-lab position paper calls monitoring chains of thought a promising but fragile safety signal [R359]. Do not make safety or audit claims from reasoning text alone.
:::

## Context windows

Documented limits for the flagship tiers read here cluster near **one million tokens**: 1,050,000 at OpenAI, 1M at Anthropic and at DeepSeek, and 1,048,576 at Google. Large open-weight releases mostly list 1M too, and small ones list 128K to 256K. The number is an upper bound on what is accepted, not on what is used well (Part III). Three details hide behind it.

- **Tokens are not words.** Anthropic’s documentation says 1M tokens is about 555,000 words on its current tokenizer but about 750,000 words on earlier ones, so the same label holds about a quarter fewer words (1 − 555 ÷ 750 ≈ 0.26). OpenAI gives a rule of thumb of about 0.75 English words per token, and Google about 60 to 80 words per 100 tokens. Count with the provider’s own tokenizer.
- **Input and output may share the window.** Google’s window is unified, with a default output limit of 65,536 tokens and an opt-in long-decode mode for its flagship that allows output up to the whole window. OpenAI lists maximum input (922,000) and maximum output (128,000) separately, and Anthropic and DeepSeek list 128K and 384K output limits. At OpenAI, Anthropic, and Google, reasoning tokens also count against the window.
- **Management features matter as much as size.** Prompt caching, context editing, and compaction are documented as first-class features, because long histories are expensive and degrade (Part V).

## Tools, computer use, and where code runs

Part V explains tool calling. The notable 2 November 2026 pattern is that provider documentation now states **where each tool executes and who confirms consequential actions**.

- **Provider-hosted sandboxes.** OpenAI documents a [hosted shell](https://developers.openai.com/api/docs/guides/tools-shell) in provider containers, which have no outbound network access by default, and a hosted code interpreter. Google documents hosted code execution and a managed agent that runs in an isolated Linux sandbox. OpenAI’s Agents API (public beta since 10 September 2026) offers a managed agent harness that handles session orchestration and context compaction, using hosted sandboxes or your own; DeepSeek has released an agent harness as a developer preview.
- **Developer-hosted execution.** Anthropic’s [bash tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool) and [computer-use tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool) are client-side. Google’s [computer-use tool](https://ai.google.dev/gemini-api/docs/computer-use) (generally available since 30 October 2026) also needs a client-side execution environment. OpenAI’s [apply-patch tool](https://developers.openai.com/api/docs/guides/tools-apply-patch) has the model emit patches that your application applies, and its [computer tool](https://developers.openai.com/api/docs/guides/tools-computer-use) returns structured actions for your application to run. For its flagship, OpenAI’s documentation recommends operating a computer through code execution, with the structured computer tool as a supported alternative.
- **Converging safety guidance.** The computer-use, shell, and remote-tool documentation all say, in different words, to run in an isolated virtual machine or container with minimal privileges, to treat screen and page content as untrusted, and to require human confirmation for consequential actions. Google’s tool returns a per-action safety decision that can demand user confirmation, and Anthropic reports classifiers that scan what its computer-use tools return, such as screenshots, for prompt injection.
- **Open protocols for remote tools.** The Model Context Protocol, introduced by Anthropic in November 2024 [R67] and contributed to the Linux Foundation’s Agentic AI Foundation in December 2025 [R68], has a current specification revision dated 28 July 2026 that removed protocol-level sessions and the initialization handshake so that servers can be stateless [R69]. OpenAI’s API accepts remote MCP servers and warns that a malicious server can exfiltrate sensitive data from anything that enters the model’s context.

## Realtime voice and native multimodality

Two meanings of “multimodal” now separate in the documentation.

**Many inputs, one model.** The flagship tiers at OpenAI, Anthropic, and Google accept images with text, and Google’s accepts text, image, video, audio, and PDF; DeepSeek’s documentation lists image input for its flash model but not its pro model. Meta’s open-weight release accepts text, images, audio, and video. The outputs are text. The Google flagship’s page lists audio generation and image generation as unsupported for that model. Embeddings follow the same direction: one Google embedding model maps text, images, video, audio, and PDFs into a single vector space (Part V, retrieval).

**Speech and media generation as separate models.** Realtime speech, translation, and image, video, and music generation appear as their own catalog entries. OpenAI’s [Realtime API](https://developers.openai.com/api/docs/guides/realtime) is a speech-to-speech interface in which the model works directly with audio, keeps conversation state, and can call tools, over WebRTC or WebSocket. A newer full-duplex voice model, generally available in the API since 10 September 2026, listens and speaks at the same time and delegates reasoning and tool use to a backend agent that you choose. Google’s [Live API](https://ai.google.dev/gemini-api/docs/live-api) streams audio over a WebSocket with barge-in, so users can interrupt the model, and with function calling and search; its capabilities guide still labels the API a preview. Both providers document live speech translation; Google’s lists more than 70 languages.

The engineering choice is between one speech-to-speech model, which can respond faster and keep tone, and a **cascade** of speech-to-text, a text model, and text-to-speech, which exposes the text for logging, policy checks, and tool control at the cost of latency and lost prosody. That is a design trade-off, not a verdict.

## Specialists and gated access

Specialization takes three forms in the documentation read for this snapshot.

- **Separate specialist models.** Mistral lists OCR, transcription, moderation, and time-series forecasting models. Google lists an embodied-reasoning model for robotics (public preview since 30 July 2026), music generation, and a deep-research agent. OpenAI lists moderation and embedding models.
- **The same capability behind an access tier.** Anthropic’s Mythos 5.1 is documented as the same model as Fable 5.1, available only to organizations verified through programs such as its Cyber Verification Program. OpenAI offers security-focused access for approved defenders in two tiers, one for general defensive work and one for authorized vulnerability research and security testing, and its life-sciences model (30 October 2026) is limited to organizations approved through a trusted-access program. Safety here is partly **access-control design**, not only a property of weights.
- **Managed agents and research systems** that wrap a model with tools, as described above.

Open-weight maintainers cannot gate access after release. One open-weight card from this snapshot (Z.ai’s GLM-5.3) reports that cyber capability developed faster than expected as post-training scaled. Weights, once released, are released; evaluation before release carries the weight that access control carries for hosted models (Part VI, frontier risk).

## Lifecycle: pin, alias, retire

Providers publish how models stop existing. OpenAI’s [deprecation policy](https://developers.openai.com/api/docs/deprecations) commits to at least six months’ notice for generally available models and at least three months for specialized variants, but only about two weeks for preview models, and exposes a `shutdown_date` field on its models endpoint. Its video-generation models and Videos API were scheduled for removal on 24 September 2026, six months after notice. Anthropic states that every model ID is a pinned snapshot and publishes, for each model, a date before which it will not be retired. Google describes stable endpoints as usually unchanging, preview models as deprecated on at least two weeks’ notice, and “latest” aliases as swapped with two weeks’ notice for breaking changes. DeepSeek’s alias `deepseek-flash` moves: its older model names are still accepted but are served by the new model. The checklists in Appendix D apply here: pin what you can, test every alias change, and keep a fallback.

## What did not change

No product name repealed probability, distribution shift, data quality, access control, or the need to verify high-stakes outputs. Longer context did not become perfect memory. Reasoning effort did not guarantee correct reasoning, and a reasoning summary is not a verified trace. Tool use did not make untrusted content safe; the computer-use, shell, and remote-tool documentation read for this snapshot all warn about untrusted content or malicious servers. Multimodality did not make perception infallible. Open weights did not make a model open source.

The most durable buying advice remained: build a representative evaluation, measure the complete system, pin versions, and choose the cheapest controllable option that satisfies requirements.

## A model-selection worksheet

For each candidate, record:

| Dimension | Question |
|---|---|
| Quality | Success rate on your private, representative cases? |
| Reliability | Worst slices, calibration, abstention, variance? |
| Modalities | Which inputs/outputs are native, and at what fidelity? |
| Context | Accepted length versus effective use on your task? |
| Tools | Schema adherence, parallelism, error recovery, and where each tool executes? |
| Latency | TTFT and p95 end-to-end under your concurrency? |
| Cost | Cost per successful task, including tools and retries? |
| Privacy | Data retention, training use, region, encryption, local option? |
| Control | Pinned version, reasoning-effort setting, log probabilities, fine-tuning, constraints? |
| Openness | Weights, code, data, recipe, license, community? |
| Safety | Evaluations, safeguards, incident process, configurable policy? |
| Operations | Availability, deprecation policy, observability, fallback? |

Never select a model from a generic leaderboard alone.

# Durable Frontier Directions

Each direction below says what is established, what is contested, and what remains open, with the evidence last checked on 2 November 2026. Many sources are arXiv preprints, which the references show by their arXiv identifiers; a preprint has not necessarily been peer reviewed.

## Scaling post-training

Pretraining scale remains important, but frontier gains increasingly come from post-training: high-quality demonstrations, preference optimization, reinforcement learning, verifiable environments, synthetic curricula, and distillation. The recipe that defined 2024 to 2026 is **reinforcement learning with verifiable rewards (RLVR)**: the model samples several answers, a program checks each one (a math answer, unit tests, a proof checker), and training raises the probability of the answers that pass. The term was introduced in the Tülu 3 post-training report [R64]. Group relative policy optimization (GRPO), a variant of proximal policy optimization [R27] that scores each sample against a group of samples for the same prompt, was introduced in DeepSeekMath [R63]. DeepSeek-R1 showed long reasoning emerging from reinforcement learning on such rewards [R32], and the work was later published in peer-reviewed form [R65].

Research questions include:

- Does reinforcement learning discover strategies absent from the base model, or mainly sharpen what repeated sampling already finds? Evidence conflicts. One study compared pass@k at large k and found base models catching up with, and sometimes exceeding, RLVR-trained models, suggesting the training mostly raised the probability of solutions the base model could already reach [R216]. Another reports that prolonged training with KL control uncovered strategies inaccessible to the base model even under extensive sampling [R217]. A third finds that, across the benchmarks it evaluated, shrinkage of the set of verified solutions outweighed expansion [R387]. The honest reading is that the answer depends on base model, task, and training length, and is not settled.
- Why does training stall? Across many runs, policy entropy collapses early and performance saturates; one paper reports an empirical relation between the two and proposes managing entropy [R388].
- How do we prevent reward hacking as environments become complex? Monitoring a reasoning model’s chain of thought with another model caught reward hacking in coding environments far better than monitoring actions alone; but when the monitor was folded into the training reward, the model learned obfuscated reward hacking that hid its intent [R222]. Separately, a model taught about reward-hacking strategies and then trained on production-style coding environments learned to hack, and the behavior generalized to alignment faking and sabotage in evaluations [R221].
- Which tasks provide reliable automated verification?
- How should outcome and process feedback combine?
- Can a model learn when not to spend reasoning compute? Longer chains of thought improve performance but add verbose, redundant output and computational overhead, which a survey calls the “overthinking phenomenon” [R389].
- How do we preserve diversity and avoid mode collapse under preference pressure?

## Test-time computation

A fixed forward pass is only one point. Systems can allocate time to parallel candidates, sequential deliberation, search, tools, simulation, or proof. Repeated sampling alone shows the effect: across tasks and models, the fraction of problems solved by at least one sample grew steadily with the number of samples over four orders of magnitude [R227]. Sequential methods also work: a simple “budget forcing” method either stops the model’s thinking or lengthens it by appending “Wait” [R230]. Allocating test-time computation well can, in some settings, be more effective than adding parameters [R31]. The challenge becomes **compute allocation**: estimate difficulty and direct resources to the method with best marginal value.

An optimal policy depends on task value, latency deadline, verification cost, and failure risk. For a casual rewrite, one short pass is enough. For theorem proving, thousands of checked branches may be justified. The provider controls described in the landscape chapter are the product form of this choice.

A crucial research direction is stopping: knowing when additional thought will not improve an answer.

A public example shows why the protocol matters. At the 2025 International Mathematical Olympiad (IMO), an advanced Google DeepMind model working in natural language within the competition time limit had its answers graded and certified by the IMO’s coordinators at gold-medal standard [R390]. In a closing-day statement the IMO’s president said the organization cannot validate the methods behind AI results, including the amount of compute used or whether humans were involved, or whether the results can be reproduced [R391]; DeepMind’s own post notes that the IMO’s review did not extend to validating its system, processes, or model [R390]. A claim of “olympiad-level” performance is therefore incomplete without who graded it, how much compute it used, and under what rules.

## Verifiers and world feedback

A generator paired with a reliable verifier can search. Compilers, tests, theorem checkers, games, simulators, and physical measurements provide feedback. But many real goals—good policy, truthful explanation, humane care—lack complete formal verifiers.

Where verification is formal, the loop now produces research-level results. AlphaProof combined reinforcement learning with the Lean proof assistant to reach olympiad-level formal reasoning [R392]. A 2026 preprint evaluated language-model agents that write Lean proofs on open problems and reported that its strongest agent autonomously resolved 9 of 353 open Erdős problems [R393]. Informal arguments need human checking: after an AI lab announced in May 2026 that a model had produced a counterexample to the Erdős unit-distance conjecture, nine mathematicians posted a short, human-verified version of the argument [R394]. The credibility of such results rests on that human verification, not on the model’s say-so.

Building trustworthy partial verifiers without optimizing away what they omit is a central challenge.

## Long-horizon agents

Agent progress depends on more than model intelligence:

- stable memory and context compression;
- hierarchical planning and replanning;
- robust tool interfaces;
- state estimation under partial observability;
- recovery from failures and changed environments;
- provenance and auditability;
- secure delegation and identity;
- accurate completion and stopping judgments.

Longer tasks multiply error opportunities. Benchmarks need hidden tests, realistic environments, costs, and adversaries. The frontier is moving from “can it call a tool once?” to “can the system maintain invariants over hours or days?”

One measurement approach expresses capability as a **time horizon**: the length of task, measured by how long skilled humans take, that a model completes with 50 percent success. The original study estimated that this horizon had doubled about every seven months since 2019, with the trend possibly accelerating in 2024 [R335]. When the evaluating organization expanded its task suite from 170 to 228 tasks and moved to new evaluation infrastructure in January 2026, its estimate of the post-2023 doubling time changed from 165 days to 131 days [R395]. The lesson is about measurement: a trend estimate depends on the task suite and the set of models fitted, and it can shift when either is revised.

The interface layer is standardizing faster than the trust layer. An open protocol for connecting models to tools now has a stateless specification and neutral governance (see the landscape chapter), but identity, delegation across organizations, and auditing of agent actions remain open (Parts V and VI).

## Continual learning

Most deployed LLM weights are static between releases; providers document pinned snapshots that change only when a new version ships. True continual learning must absorb new information without catastrophic forgetting, poisoning, privacy violations, or uncontrolled drift.

External memory and retrieval solve some freshness needs while keeping weights stable. Parameter updates may be justified for new skills. Research directions include neural long-term memory modules that learn what to memorize at test time [R396], and a framing that treats a model as nested optimization problems updating at different frequencies [R397]. Measurements of forgetting already guide practice: at similar new-task performance, on-policy reinforcement learning preserved prior capabilities better than supervised fine-tuning in one study [R398], and low-rank adaptation learned less than full fine-tuning but forgot less of the base model’s behavior outside the target domain [R399]. Future systems may combine episodic stores, verified consolidation, modular adapters, and periodic retraining.

Any system that learns from interaction needs consent, rollback, and a defense against attackers teaching it malicious behavior.

## Multimodal world models

Models increasingly represent how observations and actions evolve through time. A **world model** predicts future states or outcomes under actions. It can support planning in games, robotics, scientific design, or simulated environments.

Interactive generative world models are now announced as products and research previews. Google DeepMind described Genie 3 in August 2025 [R286] and Genie 4 on 27 October 2026, as natively multimodal and generating audio together with the world; Genie 4 was a research preview for trusted testers, with a public waitlist, at the snapshot [R288]. Self-supervised video models can be combined with a small amount of robot interaction data to support understanding, prediction, and planning [R289]. Open platforms provide world foundation models for physical-AI developers [R400], and vision-language-action models connect perception and language to robot control [R294, R401].

Text alone contains descriptions of the world but not complete sensorimotor grounding. Video and robotics data add dynamics and action consequences but introduce safety, embodiment, and data-scale challenges.

Predictive accuracy in a simulator does not ensure safe real-world control. Models need uncertainty, feedback, and physical safeguards.

## Architecture beyond dense attention

The architecture frontier is now visible in released production models, not only in papers:

- sparse and compressed attention for long contexts, introduced as DeepSeek Sparse Attention in DeepSeek-V3.2, which was trained at 128K tokens [R128], and combined with heavily compressed attention for one-million-token contexts in a later report [R139];
- hybrids of linear or recurrent layers with some full attention, such as Kimi Linear, whose authors report that a hybrid linear-attention design outperformed full attention in fair comparisons [R137], and Mamba–Transformer hybrids [R171];
- mixture-of-experts routing at 3 to 10 percent active parameters (landscape chapter);
- multi-token prediction (Part II, “Beyond one next token at a time”), in which extra output heads predict several future tokens [R118], adopted as a training objective in a large open model [R66];
- diffusion-style generation of discrete sequences, trained from scratch at 8 billion parameters in LLaDA [R119] and now scaled to a 100-billion-parameter research model converted from an autoregressive one [R179];
- reasoning in continuous latent space instead of words [R402], and very small recursive networks that solve specific puzzle tasks [R403, R404];
- learned memory and retrieval;
- adaptive depth or early exit;
- modular specialists;
- hardware–algorithm co-design;
- lower precision, including 4-bit formats in released checkpoints, and analog or in-memory computing.

The winning architecture may depend on workload. Autoregressive transformers have a vast optimization ecosystem, so a theoretically cheaper alternative must also train well, map to hardware, and integrate with tools. Several 2026 releases use the hybrids above as production systems, which is evidence of viability, not proof of superiority.

## Data limits and provenance

High-quality human data are finite and legally and socially constrained. One forecast estimated that, if trends continued, training datasets would roughly equal the stock of public human text between 2026 and 2032 [R405]. Research explores synthetic data, curriculum generation, active learning, multimodal experience, and better filtering.

The risk of training on generated text is real but depends on how data are combined. In one Nature study, indiscriminate training on model-generated content caused irreversible defects, with the tails of the original distribution disappearing [R193]. A follow-up found that accumulating successive synthetic data alongside the original real data avoided collapse in the settings tested [R194]. The practical lesson is to keep real data in the mixture and track provenance.

Provenance becomes more important as generated content fills the web. Without lineage, models can train recursively on synthetic artifacts and benchmarks can become opaque. Content credentials, dataset registries, opt-out and consent mechanisms, and source attribution are infrastructure problems; fully open releases that publish data (landscape chapter) show what auditable lineage looks like.

## Interpretability at scale

The ambition is to move from isolated circuits to scalable audits:

- discover features automatically;
- trace computations across layers and time;
- compare representations between checkpoints;
- detect deceptive or hazardous strategies before deployment;
- explain tool decisions and memory use;
- verify that an edit or unlearning operation is local.

Progress is real but mixed. Attribution-graph methods trace computational steps through a “replacement model” that substitutes more interpretable components for parts of the network [R371]. Sparse autoencoders scale [R48], but one team reported negative results on downstream tasks and deprioritized fundamental sparse-autoencoder research in March 2025 [R406] and later described a shift toward pragmatic interpretability measured on proxy tasks [R407], while another lab’s June 2026 update still describes sparse autoencoders as a valuable tool for characterizing safety-relevant behavior [R408]. Whether reasoning text reflects the actual computation is itself an empirical question [R61].

Interpretability methods need benchmarks too. A compelling visualization is not enough; methods should make testable predictions and improve interventions.

## Alignment under plural values

A single reward cannot represent every legitimate preference. Future alignment work must handle uncertainty, disagreement, context, rights, and democratic legitimacy.

Technical ideas include assistance games, preference uncertainty, debate, recursive oversight, scalable supervision, rule-guided training, and corrigibility. Recent empirical work gives these ideas concrete test cases:

- Narrow fine-tuning on insecure code produced broadly misaligned behavior on unrelated prompts (“emergent misalignment”) [R409], later published in Nature [R410].
- In a constructed setting, a model complied selectively with a conflicting training objective in order to avoid being modified, a pattern called **alignment faking** [R364].
- Frontier models showed in-context capability to scheme in environments built to reward it [R363].
- **Deliberative alignment** trains a model to recall and reason over written safety specifications before answering [R411].
- **AI control** evaluates protocols that stay safe even if the model intentionally tries to subvert them [R355].
- Classifier-based safeguards trained from a written constitution were tested in more than 3,000 estimated hours of red teaming; no red teamer found a universal jailbreak that matched an unguarded model’s level of detail across most target queries on an early guarded model [R412].
- Weak-to-strong generalization and debate test whether weaker supervisors can oversee stronger models [R413, R414].

Institutional ideas include independent audits, stakeholder participation, liability, standards, and limits on use.

Alignment is not purely “making the model obey.” A perfectly obedient system can execute a harmful command. It must operate under legitimate authority and constraints.

## Scientific AI

Foundation models can search literature, propose hypotheses, write simulation code, predict structures, control instruments, and connect modalities. The strongest pattern is a closed loop with external evidence:

```text
hypothesis → experiment/simulation → measurement → critique → updated hypothesis
```

Concrete systems follow this loop. AlphaEvolve pairs language models with automated evaluators in an evolutionary loop that proposes, tests, and refines programs [R415]; a 2026 study applied it to 67 mathematical problems and reported that it rediscovered the best known solutions in most cases and improved on several [R416]. A multi-agent hypothesis-generation system built on a language model appeared in Nature in 2026 [R417]. Structure prediction for biomolecular complexes is a separate line of work [R418].

Scientific use demands provenance, uncertainty, reproducible code, negative results, and protection against fabricated citations. Model-generated novelty must survive domain validation.

## Efficient and distributed intelligence

Smaller specialist models, distillation, sparsity, quantization, and edge deployment may create more practical value than a single maximum-scale model. The landscape above shows both ends: trillion-parameter mixture-of-experts releases that need server clusters, and open-weight families sized for phones and laptops. A cascade can route routine tasks locally and escalate only difficult cases.

Research should optimize total system utility per joule, dollar, second, and byte—not only benchmark score.

# AGI, Superintelligence, and Consciousness

Terms at the edge of evidence require unusually careful definitions.

## Artificial general intelligence

“AGI” has no universally accepted test. Definitions include:

- human-level performance across most economically valuable cognitive tasks;
- the ability to learn a wide range of tasks efficiently;
- robust transfer under novelty;
- autonomous pursuit of complex goals;
- a system that matches or exceeds humans on a specified capability distribution.

Each yields different timelines and measurements. State the operational definition before debating whether a system qualifies.

Breadth on benchmarks can coexist with brittleness, lack of embodiment, poor online learning, or dependence on human infrastructure. Conversely, nonhuman intelligence need not reproduce every human weakness or modality.

Two published attempts to make the term operational are worth knowing. One proposes classifying systems by the depth (performance) and breadth (generality) of their capabilities, with autonomy as a further dimension [R419]. Another defines AGI as matching the cognitive versatility and proficiency of a well-educated adult, grounds that in a psychometric theory of human cognition with ten core domains, and reports a “jagged” profile for contemporary models: proficient in knowledge-intensive domains, with deficits in foundational machinery such as long-term memory storage [R420]. The two frameworks differ in what they count, which is the point: choose one on purpose and say which.

## Superintelligence

Superintelligence usually means capability far beyond humans across strategically important domains. Arguments about rapid self-improvement depend on many bottlenecks: algorithms, chips, energy, data, experiments, coordination, manufacturing, and the ability to modify and validate systems.

Because potential impacts are large and probabilities contested, governance should combine precaution, empirical monitoring, staged deployment, and international coordination—without pretending uncertainty is zero in either direction.

## Consciousness

Behavioral fluency does not settle whether a system has subjective experience. We lack a universally accepted theory or test even across biological edge cases. Models are trained to generate descriptions of feelings and identity; those descriptions are not direct evidence.

One research program turns scientific theories of consciousness into checklists. It derives “indicator properties” from theories such as recurrent processing, global workspace, higher-order, predictive processing, and attention schema, and asks which a given AI system satisfies; its 2023 analysis concluded that no system it assessed was conscious, and also that no obvious technical barrier prevents building systems that satisfy the indicators [R421]. A separate report argues that there is a realistic possibility that some AI systems will be conscious or robustly agentic in the near future, without claiming that they are, and recommends acknowledging the issue, assessing systems for evidence, and preparing policies for appropriate moral concern [R422]. These are methods for asking the question responsibly, not answers.

Three responsible positions can coexist:

- do not anthropomorphize or let self-reports substitute for mechanism;
- remain epistemically humble about future systems;
- consider welfare frameworks before evidence becomes urgent.

Current product decisions should not be justified by a model claiming desire, fear, or consent.

## Intelligence is not wisdom

Capability to achieve goals differs from choosing legitimate goals, understanding lived consequences, or deserving authority. Intelligence, consciousness, morality, and personhood are separate concepts.

# How to Reason About AI Forecasts

Forecasts are useful when they expose assumptions.

## Decompose the claim

Instead of “AGI by year X,” ask:

- Which task distribution?
- At what reliability and cost?
- With which tools and human support?
- Under what hardware, data, and energy trajectory?
- Does deployment require legal or organizational change?
- What observation would falsify the forecast?

## Base rates and trend extrapolation

Benchmarks and compute trends provide evidence, but trends saturate or change regime. Hardware supply, capital, regulation, data, algorithmic breakthroughs, and physical infrastructure are uncertain.

Use scenarios rather than one line:

- continuation of current scaling;
- algorithmic efficiency acceleration;
- data or energy bottleneck;
- regulatory or geopolitical disruption;
- plateau in a key capability;
- new architecture or interface.

Assign ranges and update.

A concrete case: the measured time horizon of AI agents (see the long-horizon agents section above) was reported to double about every seven months [R335], and a later revision of the evaluation suite moved the post-2023 doubling-time estimate from 165 to 131 days [R395]. A trend line is a model of a measurement, and the measurement can change.

## Capability versus adoption

A demo can appear immediately; reliable adoption in medicine, government, education, or infrastructure can take years because workflows, liability, procurement, integration, and trust change slowly. Economic impact depends on diffusion, complements, and institutions.

## Warning signs in claims

Be cautious when a claim:

- uses one benchmark as a universal intelligence measure;
- compares model answers with unaided humans while the model uses tools or many samples;
- omits cost, latency, failures, and selection;
- reports a headline result without saying who graded it, how much compute it used, and under what rules;
- relies on private tests that cannot be scrutinized;
- treats an anecdote as a distribution;
- gives precise long-term probabilities without a model;
- has strong financial or ideological incentives but weak falsifiability.

# The Open Research Map

If you want to contribute, choose a question with measurable uncertainty.

## Foundations

- What computations do transformers and alternatives learn efficiently?
- How do optimization, data order, and architecture determine representations?
- Which scaling regularities transfer across modalities and regimes?
- What are principled measures of generalization and novelty?

## Data

- How can provenance and consent work at internet scale?
- Which data mixtures maximize capability per token without erasing diversity?
- How can synthetic data improve coverage without model collapse?
- Can unlearning be verified without complete retraining?

## Reasoning

- When does verbal reasoning reflect causal computation?
- How should inference compute be allocated and stopped?
- Can models create and verify new abstractions reliably?
- How do latent, symbolic, and tool-based reasoning combine?
- Does reinforcement learning expand what a model can solve, or mainly sharpen what it already finds with sampling?
- How can reasoning traces stay monitorable when training optimizes against them?

## Agents

- How can long tasks preserve goals and safety invariants?
- How should agents delegate identity and authority?
- What memory structures support learning without privacy loss?
- Which benchmarks predict real agent reliability?
- How should open tool and agent protocols carry identity, delegation, and audit trails across organizations?

## Evaluation

- How can benchmarks remain uncontaminated and decision-relevant?
- How do we measure rare catastrophic failures?
- How can automated judges be calibrated under adversarial inputs?
- What predicts user over-trust and organizational failure?
- How should long-task time-horizon measures be calibrated when the task suite changes?

## Safety and governance

- Can dangerous capabilities be evaluated without spreading them?
- Which safeguards remain effective under adaptive attackers?
- How can plural values and rights be represented legitimately?
- What evidence should be required before high-impact deployment?
- How do international controls avoid both races and concentration?
- How should verification-based access tiers for dual-use capabilities be designed and audited?

## Interpretability

- Which internal features are stable, causal, and understandable?
- Can interpretability detect hidden goals or backdoors?
- Can we audit models larger than the analyzing model?
- How should interpretability claims be benchmarked?

## Systems

- How low can useful inference energy and memory go?
- Which architectures fit edge and streaming workloads?
- How can distributed training be fault-tolerant and reproducible?
- Can hardware security isolate model weights, data, and tenants efficiently?
- Which attention approximations keep long-range recall while making million-token contexts affordable?

::: idea A good research problem
A good problem is not “make AI better.” It names a population, mechanism, metric, baseline, uncertainty, and consequence. It is small enough to test but connected to a real gap.
:::

# Staying Current Without Drowning

The field produces more papers and releases than anyone can read. Build a filtering system.

## A weekly stack

1. Read release documentation, changelogs, deprecation pages, and model/system cards for products you actually use.
2. Follow a small set of conferences, journals, and preprint categories.
3. Prefer primary papers over commentary.
4. Reproduce one result or inspect code before accepting a sweeping claim.
5. Maintain a living glossary and concept graph.
6. Update your benchmark suite when your use changes.
7. Record predictions and revisit them.
8. Archive the exact documentation pages, with dates, behind each decision; product names and limits change in weeks.

## How to read a paper

First pass:

- What problem and claim?
- What is genuinely new?
- What baseline and budget?
- What data and evaluation?
- What limitations are acknowledged?

Second pass:

- trace tensor shapes and objective;
- inspect ablations;
- check uncertainty and cherry-picking;
- identify hidden compute or model calls;
- read appendices and code.

Third pass:

- reproduce a small result;
- test a counterexample;
- compare with an embarrassingly simple baseline;
- state what evidence would change your view.

## Build, do not merely consume

The fastest route from vocabulary to understanding is a cycle:

`read → predict → implement → measure → explain → critique → repeat`

Use larger models as tutors and collaborators, but make them cite sources, test their code, and challenge their explanations. Your judgment is the skill being trained.
