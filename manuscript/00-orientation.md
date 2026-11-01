# Preface — What “Everything” Can Honestly Mean

You asked for a path from *nothing* to the latest artificial intelligence, and especially to large language models. That ambition is exactly right. The dishonest response would be to pretend that one finite book can hold every theorem, paper, implementation detail, product, social consequence, and unanswered question. Human knowledge is larger than any book, and the frontier moves every week.

This book makes a stronger, useful promise instead: it gives you a **connected map**. It starts without assuming calculus, programming, machine learning, or computer architecture. It builds the core ideas in dependency order, derives the transformer carefully, follows a model through data collection, pretraining, post-training, evaluation, deployment, and governance, and ends with runnable mental blueprints for building small versions yourself. It tells you what to learn next when a topic becomes an entire field.

An expert is not a person who memorized all facts. An expert has compressed many facts into models that predict what will happen, notices where those models stop working, and can locate primary evidence. The goal is therefore not trivia accumulation. The goal is to make the modern AI stack *legible*.

## The central object

A language model assigns probabilities to sequences. Given a sequence of tokens, it estimates what token could come next. A large language model, or LLM, is usually a large neural network trained on enormous collections of sequences with variants of that objective. Scale, data diversity, architecture, optimization, and post-training turn this simple statistical task into systems that can summarize, translate, write code, answer questions, call tools, interpret images and audio, and participate in multi-step workflows.

That description is true but incomplete. An actually deployed assistant is not only a neural network. It is a **sociotechnical system**:

- data sources and filtering decisions;
- tokenization and model architecture;
- accelerators, networks, compilers, and distributed training code;
- an optimization objective and training procedure;
- post-training data, preference models, and safety policies;
- a prompt assembler, retrieval layer, tool runtime, memory, and user interface;
- evaluation suites, monitoring, incident response, access controls, and governance;
- people whose work, rights, expectations, and risks shape the system.

If you study only the transformer, you understand an important engine but not the vehicle, road, traffic rules, or destination.

## Four kinds of statement

Throughout the book, keep four categories separate:

1. **Definition:** true because a term is being defined. A discrete probability distribution is a nonnegative set of weights that sums to one.
2. **Derivation:** follows from assumptions by logic or mathematics. Cross-entropy is minimized when a predicted distribution matches the data distribution, under the usual idealization.
3. **Empirical regularity:** observed repeatedly but not guaranteed. Loss often changes predictably with compute, data, and parameter scale over a range.
4. **Open interpretation:** a contested explanation or philosophical claim. Whether a model “understands” depends partly on the operational meaning of understanding.

Confusing these categories creates much AI mythology. A benchmark score is an observation under a protocol, not a definition of intelligence. A scaling curve is not a law of nature outside its measured regime. A model’s eloquent self-description is generated text, not privileged introspection.

::: warning Epistemic hygiene
A model can produce a correct explanation, an incorrect explanation with the same confidence and style, or a correct conclusion supported by an invented rationale. Treat generations as proposals whose required verification strength rises with the cost of error.
:::

## What is current, and what is durable

The technology snapshot for this **second edition** was checked on **2 November 2026**; the first edition’s snapshot was 21 August 2026. In the roughly ten weeks between them, OpenAI, Anthropic, Google, and Mistral each released new flagship-tier models, and DeepSeek replaced the model behind its flash alias (Part VIII gives dates and sources). That is the clearest evidence for the principle of this section: product names are less durable than design patterns. The book therefore emphasizes concepts that survive model releases, treats product names as dated examples, and collects the current commercial and open-weight landscape in a dated chapter near the end, where the facts are attributed to the official pages, model cards, and reports they were read on.

“Latest” should never mean “largest marketing number.” A newer model may improve one benchmark while becoming more expensive, less reproducible, more restricted, or worse for a particular language or domain. Model selection is multi-objective: quality, reliability, latency, throughput, context, modalities, privacy, control, license, energy, and total cost all matter.

## What this book does not ask you to believe

It does not ask you to believe that next-token prediction is “just autocomplete” in the dismissive sense, nor that scaling it necessarily creates a humanlike mind. It does not claim that open weights automatically imply open data or open development. It does not claim that hidden reasoning text is a faithful window into internal computation; measurements of reasoning models find that stated reasoning often omits factors that changed the answer [R61]. It does not reduce safety to refusing bad prompts, or ethics to a final checklist.

It asks you to inspect mechanisms, evidence, and trade-offs.

## A compact promise

By the final page, you should be able to:

- explain every stage from raw text to a generated response;
- read the central equations of neural language modeling;
- estimate important memory and compute costs;
- distinguish pretraining, supervised fine-tuning, preference optimization, reinforcement learning with verifiable rewards, and inference-time search;
- design a retrieval or tool-using application without confusing the wrapper with the model, and reason about the protocols, permissions, and sandboxes that connect tools and agents;
- recognize common evaluation traps and security failures;
- choose among prompting, retrieval, fine-tuning, and training from scratch;
- follow contemporary papers and documentation with a reliable conceptual scaffold;
- build a tiny transformer and an evaluation harness in principle, then know what industrial systems add;
- read a model’s documentation, license, and system card and say what each does and does not promise;
- speak precisely about uncertainty, capability, openness, risk, and governance.

That is not all knowledge. It is a route into the knowledge.

# How to Use This Book

The chapters are arranged as a dependency graph, but you do not need to read every branch at the same depth. The shortest useful path is:

1. read the one-page map and the glossary (Appendix B);
2. learn vectors, probability, derivatives, and optimization;
3. understand a neuron, backpropagation, tokenization, attention, and the transformer block;
4. follow pretraining, post-training (including preference learning, verifiable rewards, and reasoning), and generation;
5. study evaluation, reliability, and security before building an application;
6. add retrieval, tools, protocols, and agents as your application requires them;
7. choose specialist chapters and labs for your goals.

## Five reader paths

| Your goal | Read first | Then emphasize |
|---|---|---|
| Curious non-programmer | Orientation, AI map, history; Part I on learning from data; Parts II and III on tokens and attention | capabilities and limits, evaluation and reliability and the society chapter in Part VI, and the landscape and forecasting chapters in Part VIII |
| Application builder | Parts I to IV through generation, skimming derivations | Part V (context, retrieval, tools and protocols, agents, memory, multimodality), Part VI (evaluation, security, serving), Labs 6 to 11 |
| Model engineer | Parts I and II (mathematics, neural networks), then Part III (the transformer) | Part IV (data, pretraining, distributed training, post-training, verifiable rewards), inference systems in Part VI, the labs, and architecture directions in Part VIII |
| Safety, policy, and governance reader | Orientation, history, Part VI on evaluation and reliability | Part VI (security, safety and alignment, interpretability, law), Part IV on preference learning and reward hacking, Part VIII on alignment, AGI, and forecasting |
| Researcher-in-training | Entire conceptual spine | derivations, labs, bibliography, and the open research map |

## The recurring learning loop

Every difficult idea becomes easier with four passes:

- **Intuition:** What problem does the idea solve?
- **Mechanism:** What values flow through which operations?
- **Mathematics:** What is the compact precise statement?
- **Experiment:** What changes when you perturb an assumption?

For attention, intuition says “each token gathers relevant information from other tokens.” Mechanism introduces queries, keys, values, scores, masking, and a weighted sum. Mathematics writes the matrix operation. Experiment changes the mask or temperature and inspects the result. None of the passes should replace the others.

## Notation without fear

Symbols are compressed sentences. Read them slowly.

- A lowercase scalar such as `x` is one number.
- A bold lowercase vector such as **x** is an ordered list of numbers.
- An uppercase matrix such as **W** is a rectangular table of numbers and usually a linear transformation.
- A subscript, as in `x_t`, selects an item, often at time or token position `t`.
- A superscript can identify a layer or exponent; context will say which.
- `Σ` means “sum these terms.”
- `∂L/∂w` asks how sensitively loss `L` changes when weight `w` changes.
- `E[X]` is an average weighted by a probability distribution.
- `≈` means approximately, not exactly.
- “(computed)” after a figure means the book derived it by arithmetic from numbers stated nearby, so you can check it yourself.

When an equation appears, ask: What are the shapes? What is fixed? What is learned? What is the output used for? Shape checking catches many mistakes before algebra does.

## Active reading questions

At the end of any section, close the page and answer:

1. What problem was being solved?
2. What information enters and leaves the mechanism?
3. What does training change?
4. Which assumptions could fail?
5. How would I test the claim?

If you cannot answer, reread for the missing connection rather than memorizing phrases.

## A safe experimentation ladder

Start with tiny, observable systems. A ten-token vocabulary reveals softmax behavior better than an API with hidden internals. A one-layer transformer reveals masking better than a billion-parameter checkpoint. Move upward only when the added scale answers a question.

A sensible ladder is:

- calculate probabilities by hand;
- implement scalar automatic differentiation;
- train a small multilayer perceptron;
- tokenize a miniature corpus;
- implement one attention head;
- train a tiny character- or subword-level transformer;
- adapt an existing small open-weight model;
- build retrieval and a tool loop;
- evaluate, attack, monitor, and document the complete application.

Never place secrets, private data, or unreviewed autonomous permissions into an experiment. Use sandboxes, spending caps, timeouts, allowlists, and human confirmation for consequential actions.

::: practice Feynman checkpoint
Explain an LLM to an imaginary twelve-year-old using only the words *pieces*, *numbers*, *patterns*, *guess*, and *feedback*. Then explain it to an engineer using tokens, tensors, conditional probability, gradient descent, and systems constraints. The gap shows what you still need to connect.
:::

# The One-Page Map of Artificial Intelligence

Artificial intelligence is the broad project of building artifacts that perform tasks associated with perception, prediction, language, reasoning, planning, learning, creativity, or action. The boundary moves: once a technique becomes routine, people often stop calling it AI.

## A useful taxonomy

**Symbolic AI** represents knowledge and procedures explicitly: logic, rules, search trees, planners, theorem provers, knowledge graphs, and constraint solvers. It can be exact and inspectable, but acquiring rules and handling ambiguity are hard.

**Machine learning** fits behavior from data rather than hand-writing every rule. Supervised learning uses labeled examples, unsupervised learning seeks structure, self-supervised learning creates targets from the data itself, and reinforcement learning uses consequences or rewards.

**Deep learning** uses neural networks with many composed layers. Rather than manually choosing every feature, training discovers internal representations useful for an objective.

**Generative modeling** learns a distribution well enough to create new samples: text, images, audio, video, molecules, code, or combinations.

**Foundation models** are broadly trained models adapted to many downstream tasks. An LLM is a foundation model centered on language-like token sequences, although current systems may accept and produce multiple modalities.

**Agents** are systems that use a model inside a loop: observe state, decide, call tools or act, receive results, update context or memory, and continue until a stop condition. Agency belongs to the whole loop and permission structure, not to the text generator alone.

## The modern stack in one flow

```text
WORLD AND PEOPLE
  ↓ collection, consent, licensing, sensors, documents, code, media
RAW DATA
  ↓ cleaning, deduplication, filtering, mixing, annotation
TRAINING EXAMPLES
  ↓ tokenization / modality encoders
INTEGER IDS AND TENSORS
  ↓ large-scale optimization of prediction objectives
BASE MODEL
  ↓ instruction data, preferences, rewards, safety and tool training
POST-TRAINED MODEL
  ↓ prompts + policies + retrieval + memory + tools + decoding
APPLICATION
  ↓ evaluation, monitoring, feedback, incidents, governance
REAL-WORLD CONSEQUENCES
  ↺ new data and decisions feed back into the system
```

Each arrow contains technical and political choices. Removing duplicated text changes memorization and weighting. A tokenizer changes cost across languages. A reward specification changes which behavior gets reinforced. A user interface changes how much authority people grant an answer.

## Training versus inference

**Training** changes model parameters. It processes examples, computes a loss, propagates gradients, and updates weights. Pretraining can require enormous one-time computation.

**Inference** keeps parameters fixed and computes outputs for new inputs. In autoregressive generation, the model runs repeatedly, appending one selected token at a time. Inference may also perform search, sample several candidates, use tools, retrieve documents, or verify answers; these add compute without changing the base weights.

**Learning in context** usually means behavior changes because examples or instructions are present in the prompt. The parameters do not change. **Fine-tuning** does change parameters. **External memory** changes stored information outside the parameters. These mechanisms are often conflated.

## Capability is not reliability

A capability asks whether the system can succeed under some conditions. Reliability asks how often it succeeds across the real distribution, under shifts, attacks, ambiguity, and operational constraints. A model that solves one hard problem may still fail an easy paraphrase. A system can be impressive and unsuitable for unsupervised use at the same time.

Think in distributions rather than anecdotes:

`expected utility = Σ probability(outcome) × value(outcome)`

Then inspect the tails. Rare high-cost failures can dominate expected risk.

## The three boundaries

Every AI system has at least three boundaries:

1. **Knowledge boundary:** what evidence and training can support.
2. **Competence boundary:** which transformations it performs reliably.
3. **Authority boundary:** what it is permitted to decide or do.

Good engineering makes all three visible. Retrieval can move the knowledge boundary. Better training can move competence. Access controls and human review define authority. No benchmark should silently expand authority.

# A Compressed History: Ideas That Converged

Modern LLMs did not appear from a single breakthrough. They combine centuries of work on logic and probability, decades of statistics and computer science, improvements in semiconductor manufacturing and distributed systems, vast digital corpora, and particular social and economic conditions.

## Mathematical roots

Formal logic asked which conclusions follow from premises. Probability theory quantified uncertainty. Calculus described change and enabled optimization. Linear algebra represented transformations in many dimensions. Information theory later measured coding and uncertainty. These are not preliminary trivia; they remain the language of current models.

## Computation and early AI

In 1936, Alan Turing formalized a simple abstract machine capable of expressing general algorithms. In 1950 he reframed “Can machines think?” as an operational imitation game [R01]. The 1956 Dartmouth workshop helped establish artificial intelligence as a field.

Early programs searched spaces of symbolic states, proved theorems, solved constrained puzzles, and manipulated rules. Optimism collided with combinatorial explosion, brittle representations, limited data, and limited compute. The lesson was not that symbols were useless. Search, planning, code execution, and formal verification remain essential—and increasingly complement neural models.

## Neural learning arrives, stalls, and returns

McCulloch and Pitts proposed a mathematical neuron in 1943. Rosenblatt’s perceptron learned a linear decision boundary. Its inability to represent some simple functions with one layer became famous; multilayer networks could solve them, but training deep compositions was difficult.

Backpropagation efficiently applied the chain rule to compute gradients through layered networks and became widely influential after Rumelhart, Hinton, and Williams in 1986 [R02]. Neural methods still went through cycles because datasets, initialization, algorithms, and hardware were not ready for today’s scale.

## Statistical language modeling

Language modeling long predated transformers. An n-gram model estimates the next word from a short fixed history by counting. It is interpretable and useful, but sparse: most plausible sequences never appear in a finite corpus.

Neural language models replaced exact context counts with learned continuous representations. Bengio and colleagues showed a feed-forward neural probabilistic language model in 2003 [R03]. Distributed word vectors let similar words share statistical strength. Word2vec later made efficient embedding learning widely accessible [R04].

Recurrent neural networks carried a hidden state through a sequence. Long short-term memory (LSTM) added gates to preserve or forget information. Sequence-to-sequence models mapped one sequence to another [R05]. Attention then let a decoder look directly at different encoder states rather than compress everything into one fixed vector [R06].

## Deep learning’s practical turn

Around the 2010s, several trends aligned:

- GPUs made dense linear algebra much faster;
- large labeled and unlabeled datasets became available;
- improved activations, initialization, normalization, and optimizers stabilized training;
- software frameworks automated differentiation and accelerator execution;
- benchmark gains attracted capital and talent.

Convolutional networks transformed image recognition. Recurrent and attention-based systems improved speech and translation. Reinforcement learning combined deep networks with search in games.

## The transformer and scale

The 2017 paper “Attention Is All You Need” replaced recurrence in the core sequence model with stacked self-attention and feed-forward blocks [R07]. Parallel processing made training scale efficiently. Encoder models such as BERT learned bidirectional representations [R08]. Decoder-only models such as GPT learned autoregressive generation and increasingly broad task behavior [R09]. Encoder–decoder models such as T5 framed many tasks as text-to-text [R10].

Scaling studies found smooth relationships among model size, data, compute, and predictive loss within observed regimes [R11]. Later work emphasized that parameters and training tokens must be balanced; an oversized model trained on too little data is compute-inefficient [R12]. Scale did not remove the need for good data or algorithms, but it made a single pretrained model adaptable to many tasks.

## From completion to assistants

A base model predicts continuations; it does not inherently know the intended role, truth standard, or safety policy. Instruction tuning trained models on demonstrations of tasks. Reinforcement learning from human feedback (RLHF) fitted preference signals and optimized behavior [R13]. Direct preference optimization (DPO) later expressed a related preference objective without a separate online reinforcement-learning loop [R14]. Constitutional and AI-assisted feedback explored scalable rule-guided supervision [R15].

Tool use, retrieval, structured outputs, multimodal encoders, long contexts, and inference-time search turned a standalone generator into a component inside larger systems. Mixture-of-experts architectures activated only part of a large parameter set for each token. Quantization and optimized serving made capable models cheaper to run.

## Reasoning, agents, and open ecosystems, 2024–2026

Four developments after the instruction-following assistant changed what “a model” means. Each is taught in full later; here they are placed in time.

**Reasoning models trained with reinforcement learning.** In September 2024 OpenAI described a model trained with reinforcement learning to reason at length before answering [R62]. The ingredients were published openly: group relative policy optimization (GRPO) in a February 2024 mathematics paper [R63], the term “reinforcement learning with verifiable rewards” in a November 2024 post-training report [R64], and, in January 2025, DeepSeek-R1 [R32], later published in peer-reviewed form [R65]. The idea is to reward answers that a program can check, so that spending more computation while thinking pays off on hard problems. By late 2026 hosted providers expose how hard the model thinks as a setting (Part VIII).

**Open-weight mixture-of-experts models at frontier scale.** DeepSeek-V3, reported in December 2024, was a mixture-of-experts model with 671 billion total parameters and 37 billion activated for each token [R66]. By the 2 November 2026 snapshot, open-weight releases of roughly one to 2.8 trillion total parameters, with only a few percent active per token, were available under licenses ranging from MIT to custom terms (Part VIII).

**Agents and protocols.** Models increasingly act in loops: observe, call a tool, read the result, continue. Standards for the connections appeared. Anthropic introduced the Model Context Protocol, an open standard for connecting models to tools and data, in November 2024 [R67]; it was contributed to the Linux Foundation’s Agentic AI Foundation in December 2025 [R68]; and its specification revision of 28 July 2026 made servers stateless [R69]. Providers now also offer managed runtimes with sandboxes, context compaction, and permission controls (Parts V and VIII).

**Long context, multimodality, and specialization.** Context windows of about one million tokens became common on flagship models; audio, image, and video understanding and live voice entered mainstream interfaces; and specialist or access-restricted models appeared for security, life sciences, robotics, and documents (Part VIII).

## The current convergence

By the mid-2020s—and continuing into the 2 November 2026 snapshot—the dominant frontier pattern was not merely “a bigger chatbot.” It was a family of interacting capabilities:

- text, image, audio, and sometimes video understanding, with generation of audio, images, and video often handled by separate models;
- adjustable reasoning effort, increasingly with adaptive thinking;
- code execution, search, computer use, and external tools connected through open protocols;
- long-running agent workflows with state, permissions, and managed sandboxes;
- sparse and specialized architectures (mixture of experts, linear and sparse attention) for efficiency;
- open-weight ecosystems spanning phones to trillion-parameter clusters;
- synthetic data, model-generated feedback, and verifiers;
- specialist and access-restricted models;
- stronger emphasis on evaluations, system cards, security, and regulation.

The boundaries between model, runtime, product, and institution became more important—not less.

## Recurring lessons from the history

1. **No single paradigm owns intelligence.** Statistical learning, search, tools, memory, logic, and human organization can be complementary.
2. **Scale changes which methods work.** An algorithm’s practical value depends on hardware, data, and software.
3. **Benchmarks become targets.** Once optimized aggressively, they lose some diagnostic value.
4. **Abstraction hides labor.** Datasets, annotation, content moderation, chip fabrication, and operations remain human systems.
5. **Predictions should be conditional.** “If compute, data, energy, and investment continue under these constraints…” is more meaningful than an unconditional date for general intelligence.
6. **Old problems return at new scale.** Distribution shift, specification, security, uncertainty, and accountability do not disappear when outputs become fluent.
7. **Names expire faster than patterns.** Most flagship model names in this edition’s snapshot are newer than those in the first edition’s, while the patterns they share—reasoning effort, tools, long context, sparse architectures—were already visible.

::: idea Mental model
The history of AI is not a straight march from symbolic systems to neural systems. It is a growing toolbox whose components become newly useful when interfaces, scale, and economics change.
:::
