## Literature Review: Toward Layered Latent Memory Systems

The research problem is no longer simply “how do we retrieve more context?” Recent work is converging on a larger question:

> How should an LLM agent remember, update, retrieve, consolidate, and forget across a long stream of documents, conversations, tool calls, corrections, and changing world state?

This proposal sits between three poles:

1. **Explicit memory**: retrieval, documents, citations, source metadata.
2. **Parametric memory**: knowledge stored in model weights or adapter weights.
3. **Latent operational memory**: learned internal modules that route, compress, consolidate, and decide where to look.

The goal is not to replace retrieval with weights, or to replace weights with retrieval. The goal is a hybrid memory stack: exact evidence remains explicit, while fuzzy memory, temporal scoping, retrieval policy, and consolidation are learned in latent space.

---

## 1. Explicit Memory: RAG, REALM, RETRO

The classic retrieval-augmented line starts from a practical limitation: pretrained language models store factual knowledge in parameters, but precise access, updating, and provenance are difficult. RAG introduced a generator conditioned on retrieved documents from a dense index, explicitly combining parametric and non-parametric memory for knowledge-intensive generation. REALM extended this idea by pretraining with a latent document retriever, arguing that retrieval makes knowledge access more modular and interpretable than forcing all facts into weights. RETRO scaled retrieval further by conditioning on chunks from a massive external database, showing that explicit memory can substitute for some parameter count while improving factual access. ([arXiv][1])

**Key idea for this proposal:** retrieval is the right substrate for exact evidence, provenance, deletion, and source-grounded answers.

**Limitation:** RAG alone does not give the model a durable fuzzy understanding of a project or domain. It may retrieve the right chunk, but it does not necessarily learn what changed, which source is stale, which prior plan was abandoned, or when it should retrieve before answering.

---

## 2. Retrieval-Policy Training: ReAct, Toolformer, Self-RAG, RAFT

A second branch says that retrieval is not enough; the model needs to learn how to use memory and tools.

ReAct interleaves reasoning and actions, allowing models to call external sources while solving tasks. Toolformer trains models to decide when to call APIs, what arguments to pass, and how to incorporate results. Self-RAG trains models to retrieve adaptively and critique retrieved passages and answers. RAFT trains models for open-book, domain-specific QA by teaching them to use relevant retrieved documents while ignoring distractors. These works support the idea of a **protocol LoRA**: a stable adapter whose job is not to memorize the corpus, but to practice correct retrieval/tool behavior.

**Design implication:**

```text
Protocol LoRA:
  learn when to retrieve
  learn how to query
  learn when to cite
  learn when to defer
  learn how to ignore distractors
  learn how not to trust stale parametric priors
```

For the proposed system, this is central. The “dreaming” loop should not just train latent memory to answer from memory; it should continue practicing retrieval discipline.

---

## 3. LoRA and QLoRA: Cheap Modular Adaptation

LoRA freezes a pretrained model and injects low-rank trainable matrices into selected transformer layers. A frozen linear projection becomes:

```text
y = W x + B A x
```

QLoRA makes this practical on consumer hardware by backpropagating through a frozen 4-bit quantized model into LoRA adapters. This matters because most early experiments should be possible on a personal 3090-class GPU. ([arXiv][2])

**Key idea for this proposal:** LoRA is the easiest experimental scaffold for testing modular memory behavior.

But LoRA should be separated into roles:

```text
Protocol LoRA:
  stable memory-use behavior

Temporal LoRA:
  time, staleness, source-scope behavior

Fast/slow LoRA:
  first experimental proxy for memory plasticity

LoRA-as-memory:
  possible, but maybe not the final substrate
```

This distinction matters because recent work suggests vanilla LoRA is not necessarily the best substrate for continual memory.

---

## 4. LoRA as Knowledge Memory

A very recent anchor paper is **Understanding LoRA as Knowledge Memory: An Empirical Analysis**. It explicitly studies LoRA as a modular parametric memory, examining storage capacity, internalization, composability, multi-module scaling, and long-context reasoning. The paper positions LoRA as a complementary memory axis alongside RAG and in-context learning. ([arXiv][3])

This directly supports the research framing:

> LoRA can be studied not merely as “fine-tuning,” but as a modular knowledge-memory object.

However, the same framing raises important questions:

```text
How much can a LoRA store?
When does it generalize?
When does it blur contradictions?
How do multiple LoRAs compose?
Can LoRA memory remain editable?
Can LoRA memory coexist with retrieval?
```

**Design implication:** LoRA is a serious baseline for memory, but should be tested against sparse memory layers and retrieval-grounded systems.

---

## 5. LoRA Composition, Stacking, and Adapter Reuse

The question “can we stack LoRAs?” has multiple answers. Additive stacking is straightforward:

```text
y = W x + ΔW_protocol x + ΔW_temporal x + ΔW_memory x
```

The harder question is whether stacked adapters compose without interference.

LoraHub studies dynamic composition of LoRA modules for unseen tasks using a few examples, treating LoRAs as reusable task modules. AdaMix uses a mixture of adaptation modules, including LoRA-style modules, inside transformer layers. More recent work such as SCALE-LoRA treats open-pool LoRA reuse as a post-retrieval composition problem: retrieve candidate adapters, compose them, and audit disagreement/reliability. TC-LoRA addresses task interference by clustering data and factorizing LoRA adapters into shared and task-specific components. ([arXiv][4])

**Design implication:** stacking LoRAs is possible, but naive stacking is not enough. The experimental design should compare:

```text
single rank-matched LoRA
additive stacked LoRAs
gated stacked LoRAs
retrieved/composed LoRAs
orthogonalized LoRAs
sparse memory layers
```

---

## 6. Continual LoRA: Forgetting, Orthogonalization, and Merging

Sequential LoRA training can still forget. **Low-Rank Continual Personalization of Diffusion Models** shows that naive continual LoRA fine-tuning causes interference and forgetting, while sequential merging, orthogonal initialization, and magnitude-based selection reduce forgetting in diffusion personalization. Although the domain is diffusion, the lesson transfers: low-rank adapters are modular, but sequential low-rank updates still interfere. ([arXiv][5])

**DualLoRA** introduces orthogonal and residual LoRA components to balance stability and plasticity in continual learning. **Merge before Forget** orthogonally initializes and sequentially merges LoRA updates into a single unified LoRA, using time-aware scaling to balance old and new knowledge. ([arXiv][6])

**Design implication:** the first LoRA memory experiments should not simply “keep training one LoRA forever.” Better baselines include:

```text
continual single LoRA
fast/slow LoRA
orthogonal fast/slow LoRA
time-aware LoRA merging
LoRA + replay
LoRA + distillation
rank-matched single LoRA
```

---

## 7. Routed LoRA and LoRA-MoE

A strong recent direction is to treat LoRA modules as experts.

LoRAMoE adds multiple LoRA adapters and a router to reduce world-knowledge forgetting during instruction tuning. MixLoRA builds a sparse MoE from LoRA experts inside a frozen dense model. ELDER uses multiple LoRAs and a router for lifelong model editing, explicitly trying to create smooth associations between semantically similar inputs and adapter allocations. AM-LoRA learns an attentional mixture over LoRAs for continual learning. LD-MoLE learns dynamic token-dependent and layer-wise routing over LoRA experts, allowing the number of activated experts to vary. MoLoRA routes adapters per token, which is useful when one request needs multiple skills. ([arXiv][7])

This supports the “latent router graft” idea:

```text
hidden state h
  -> router(h)
  -> weights over LoRA experts
  -> apply protocol / temporal / fast / slow / domain adapters
```

A possible layer equation:

```text
y = W x
  + g_protocol(h) ΔW_protocol x
  + g_temporal(h) ΔW_temporal x
  + g_fast(h) ΔW_fast x
  + g_slow(h) ΔW_slow x
```

**Design implication:** a dense model plus routed LoRA bank is a better first experiment than a full MoE model. It tests the latent routing idea without adding MoE training confounds.

---

## 8. LoRA Systems: Runtime Cost and Adapter Switching

Dynamic LoRA routing is not free. LoRA-Switch finds that dynamic adapters can incur large inference latency overhead due to fragmented CUDA kernels, and proposes fused switching to reduce decoding latency. Activated LoRA focuses on efficient multi-adapter serving with cross-model KV-cache reuse, reporting large latency reductions when switching between base and adapted models in multi-turn pipelines. ([arXiv][8])

**Design implication:** if the system stacks or routes many LoRAs, serving efficiency must be measured, not assumed.

Experiments should track:

```text
adapter count
active adapters per token / sequence
tokens/sec
KV-cache reuse
latency
VRAM footprint
reload/hotswap overhead
```

---

## 9. Representation Grafts: ReFT and Hidden-State Interventions

LoRA modifies effective weights. A different graft style modifies hidden representations directly.

ReFT freezes the base model and learns interventions on hidden states rather than weight matrices. LoReFT is a low-rank representation intervention that can be more parameter-efficient than LoRA. CS-ReFT composes multiple representation subspace edits with a lightweight router to reduce cross-skill interference. ([arXiv][9])

This is close to the “side graft” idea:

```text
h_l = hidden state at layer l

h_l' = h_l
     + g_1(h_l) Intervention_1(h_l)
     + g_2(h_l) Intervention_2(h_l)
     + g_3(h_l) Intervention_3(h_l)
```

**Design implication:** some memory operations may be better as representation grafts than LoRA weight deltas:

```text
temporal filtering
stale-prior inhibition
source-scope routing
confidence / deferral steering
```

This suggests an ablation:

```text
temporal LoRA
vs
temporal ReFT graft
vs
temporal-gated sparse memory
```

---

## 10. Hypernetwork-Generated LoRA and Memory Compiler Grafts

The most aggressive version of the proposal is a memory compiler: given a messy corpus, generate initialization, gates, memory slots, or even adapter weights.

SHINE maps context into LoRA adapters in a single forward pass, turning context into in-parameter knowledge without ordinary fine-tuning. Zhyper uses a factorized hypernetwork to generate context-aware LoRA adapters from textual descriptions. ([arXiv][10])

This matches the proposed **memory compiler graft**:

```text
messy corpus
  -> compiler graft
  -> temporal coordinates
  -> source weights
  -> conflict clusters
  -> memory slots
  -> adapter/graft initialization
```

**Design implication:** hypernetwork-generated adapters are probably not the first experiment, but they are the natural long-term version of “learn how to initialize memory from a corpus.”

---

## 11. Sparse Memory Layers and Continual Learning

The most relevant recent paper for the memory substrate is **Continual Learning via Sparse Memory Finetuning**. It argues that catastrophic forgetting arises partly because trainable parameters are shared across too many tasks. It uses memory-layer models and updates only memory slots that are highly activated by new knowledge relative to background usage. The paper reports much less forgetting than full fine-tuning or LoRA at the same new-knowledge acquisition level: NaturalQuestions F1 drops by 89% after full fine-tuning, 71% with LoRA, but only 11% with sparse memory finetuning. ([arXiv][11])

This shifts the architecture:

```text
LoRA:
  best for protocol / temporal behavior

Sparse memory layer:
  better candidate for fuzzy continually updated memory

Retriever:
  exact evidence and provenance
```

**Design implication:** LoRA should remain an important baseline, but the serious memory substrate may be sparse and slot-like.

A revised experiment ladder:

```text
Base RAG
Protocol LoRA
Continual LoRA
Fast/slow LoRA
Temporal-gated LoRA
Sparse memory finetuning
Sparse memory + protocol LoRA
Sparse memory + protocol LoRA + retrieval practice
```

---

## 12. Neural Memory Modules: Titans, ATLAS, TNT

A newer architecture family treats memory as an explicit neural module rather than retrieval or adapter weights.

Titans introduces a neural long-term memory module and frames attention as short-term memory while neural memory acts as persistent long-term memory. ATLAS critiques recurrent long-term memory modules for limited capacity and online-only updates, then proposes a high-capacity memory module optimized using current and past tokens. TNT improves training efficiency for test-time memorization systems by separating global and local memory modules and then fine-tuning local modules. ([arXiv][12])

This literature supports the idea that the memory module should not merely be “more context.” It can be a trainable latent state with its own update rules.

**Design implication:** a separate memory module is plausible, but the first experiments should be smaller:

```text
LoRA proxy memory first
sparse memory layer second
separate neural memory module third
```

---

## 13. Memory as a Separate Model: MeMo, Mem-π, MemoBrain

MeMo, the paper that sparked this thread, encodes new knowledge into a dedicated memory model while keeping the LLM unchanged. It is not primarily a latent graft inside the base model; it is closer to memory as an external specialist model. Its claimed advantages include cross-document relationship modeling, robustness to retrieval noise, avoidance of catastrophic forgetting in the main LLM, compatibility with closed-source LLMs, and inference cost independent of corpus size. ([arXiv][13])

Mem-π also uses a separate model, but instead of retrieving static memory entries, it generates context-specific guidance on demand and learns when to abstain. MemoBrain proposes an executive memory model for tool-augmented agents, organizing reasoning steps, pruning invalid steps, folding completed sub-trajectories, and preserving a compact reasoning backbone under context limits. ([arXiv][14])

These papers form a foil for the graft proposal:

```text
MeMo / Mem-π / MemoBrain:
  memory as an external specialist model

Layered latent graft proposal:
  memory as internal adapter/graft behavior + sparse latent memory + explicit retrieval
```

**Design implication:** compare against memory-as-model systems as a serious alternative. They may be more compatible with closed models, while grafts may offer tighter integration with the base model’s hidden states.

---

## 14. Agent Memory Benchmarks: MemoryAgentBench and MemGym

MemoryAgentBench argues that memory agents need four capabilities: accurate retrieval, test-time learning, long-range understanding, and selective forgetting. It evaluates memory through incremental multi-turn interactions rather than static long-context QA. ([arXiv][15])

MemGym is even closer to the intended project setting. It argues that existing benchmarks overfocus on personalized chat and miss memory formation during long agent execution. It includes tool-use dialogue, deep-research search, coding, and computer-use regimes, and reports memory-isolated scores that decouple memory quality from reasoning, retrieval, and tool-use ability. ([arXiv][16])

**Design implication:** Claude Code / Codex logs are not just data; they are a realistic benchmark source.

Possible evaluation tasks:

```text
retrieve relevant prior session
identify stale plan vs current implementation
predict relevant files
preserve project conventions
remember user corrections
choose whether to inspect repo or retrieve memory
answer with citations to prior conversations / files
```

---

## 15. Hierarchical and Modular Agent Memory

H-MEM proposes hierarchical memory for LLM agents, organizing memory by semantic abstraction and using index-based routing layer by layer. A 2026 survey, **Memory in the LLM Era**, frames memory as a core module for long-horizon LLM agents and compares representative memory methods under a unified framework. ([arXiv][17])

**Design implication:** memory should not be a flat vector store. It likely needs levels:

```text
episode-level traces
project-level summaries
stable conventions
tool-use policies
source-grounded facts
latent fuzzy state
```

This supports the “fuzzy remember where to look” framing:

> Memory should often remember the region of experience where evidence lives, not the exact answer.

---

## 16. Temporal Representation and Learning “Time”

The system should not hand-code time as a simple date. “Time” in memory means recency, validity, volatility, source authority, narrative order, staleness, planning-vs-canon, and current-vs-historical query intent.

Time2Vec learns vector representations of time using periodic and non-periodic components. Time-aware language models train on text with timestamps to represent facts as temporally scoped. Temporal KG work such as Know-Evolve and diachronic embeddings treats entity representations as time-dependent. Temporal Graph Networks maintain memory over event streams. ([arXiv][18])

A temporal-gated LoRA experiment can test whether the model can learn “memory-time”:

```text
z_time = TemporalEncoder(prompt_or_chunk)

y_l = W_l x_l + g_l(z_time) * LoRA_l(x_l)
```

**Design implication:** compare manual fast/slow LoRA against learned temporal-gated LoRA.

---

## 17. Temporal QA and Evolving Knowledge Benchmarks

TemporalWiki evaluates models across consecutive Wikipedia/Wikidata snapshots, testing whether they acquire updated knowledge while retaining old knowledge. StreamingQA evaluates QA over timestamped news articles across many years. PAT-Questions focuses on present-anchored questions where answers depend on the current date. ([arXiv][17])

The exact benchmark set needs cleanup in the post, but the task categories are clear:

```text
current-state QA
historical as-of QA
what-changed QA
stale-fact suppression
retention of unchanged facts
source-grounded answers
contradiction resolution
```

---

## 18. Neuroscience and Cognitive Memory

The neuroscience grounding is strongest as a design prior, not as a literal module mapping.

Complementary Learning Systems theory distinguishes fast hippocampal-like learning from slow cortical-like learning. Benna and Fusi model memory as multiple interacting processes over different timescales, where memories enter fast variables and transfer into slower variables. Continual-learning papers inspired by CLS similarly explore fast and slow learning systems. Hippocampal sharp-wave ripples and replay are associated with consolidation during rest and sleep; predictive coding frames the brain as maintaining a generative model updated by prediction error. ([arXiv][19])

**Useful analogy:**

```text
context window:
  working memory

retriever:
  exact external evidence

fast memory:
  recent episodic traces

slow memory:
  consolidated semantic structure

dreaming loop:
  replay / consolidation

router:
  attention / control / gating
```

**Important caveat:** LoRA is not literally cortex, and a sparse memory layer is not literally hippocampus. Neuroscience motivates fast/slow consolidation and replay, but the artificial modules need empirical validation.

---

## 19. The Central Synthesis

The current literature suggests a revised architecture:

```text
Base dense LLM:
  frozen general prior

Protocol LoRA:
  memory/tool/retrieval discipline

Temporal LoRA or ReFT graft:
  source scope, staleness, current-vs-historical reasoning

Sparse memory layer:
  fuzzy continually updated latent memory

Retriever:
  exact evidence and provenance

Dreaming loop:
  offline replay over conversations, docs, tool calls, corrections

Memory compiler:
  later-stage meta-learner that initializes memory from messy corpora
```

The proposal should no longer imply that LoRA alone is the final memory substrate. A better claim is:

> LoRA is the easiest scaffold for studying memory behavior, but sparse memory layers and representation grafts may be better substrates for continually updated fuzzy memory.

---

## 20. Experimental Ladder

The blog should build toward the dream in stages:

```text
1. Base RAG
2. Protocol LoRA for retrieval/tool traces
3. Continual single LoRA
4. Fast/slow LoRA
5. Orthogonal fast/slow LoRA
6. Learned temporal-gated LoRA
7. Routed LoRA bank
8. ReFT-style temporal representation graft
9. Sparse memory finetuning
10. Sparse memory + protocol LoRA + retrieval
11. Dream replay over conversations/docs/tool traces
12. Memory compiler / hypernetwork initialization
13. Optional MoE memory expert
```

The most important early comparison:

```text
single rank-matched LoRA
vs
fast/slow LoRA
vs
temporal-gated LoRA
vs
sparse memory finetuning
```

If temporal-gated LoRA does not beat a rank-matched single LoRA, the layered-LoRA part is probably overbuilt.

If sparse memory finetuning beats LoRA by a large margin, then LoRA should become the protocol layer, not the primary memory substrate.

---

## 21. Updated Research Claim

A stronger, current version of the proposal is:

> Recent work suggests LLM memory is becoming a first-class module rather than a prompt trick. RAG preserves exact evidence but lacks fuzzy consolidation; LoRA enables cheap modular adaptation but can still interfere under continual updates; sparse memory layers reduce forgetting by sparsely updating only relevant slots; and memory-as-model approaches show that dedicated memory components can capture cross-document relationships. We propose a layered hybrid system where a protocol/temporal LoRA teaches memory-use behavior, sparse or grafted latent memory stores fuzzy evolving state, and explicit retrieval preserves provenance. Offline replay over conversations, documents, and tool traces updates memory while continuing to practice correct retrieval.

---

## 22. Short Bibliography Clustered by Use

### Explicit retrieval / evidence memory

* RAG — retrieval plus parametric generation. ([arXiv][1])
* REALM — retrieval-augmented pretraining with modular knowledge access. ([arXiv][20])
* RETRO — large-scale retrieval-enhanced transformer. ([arXiv][21])

### Tool and retrieval behavior

* ReAct — interleaved reasoning and acting.
* Toolformer — learned API/tool calls.
* Self-RAG — adaptive retrieval and self-critique.
* RAFT — retrieval-augmented fine-tuning with distractors.

### LoRA and PEFT

* LoRA — low-rank trainable deltas in frozen models. ([arXiv][2])
* QLoRA — 4-bit frozen base plus trainable LoRA. ([arXiv][22])
* Understanding LoRA as Knowledge Memory — LoRA as modular parametric memory. ([arXiv][3])

### LoRA composition and stacking

* LoraHub — dynamic LoRA composition. ([arXiv][4])
* AdaMix — mixture of adaptation modules. ([arXiv][23])
* SCALE-LoRA — post-retrieval LoRA composition and reliability auditing. ([arXiv][24])
* TC-LoRA — clustered/tensorized LoRA merging to reduce interference. ([arXiv][25])

### Continual LoRA

* Low-Rank Continual Personalization — adapter merging/orthogonalization to reduce forgetting. ([arXiv][5])
* DualLoRA — orthogonal + residual LoRA for stability/plasticity. ([arXiv][6])
* Merge before Forget — time-aware sequential LoRA merging. ([arXiv][26])

### Routed LoRA / LoRA-MoE

* LoRAMoE — LoRA router to reduce world-knowledge forgetting. ([arXiv][7])
* MixLoRA — sparse MoE from LoRA experts. ([arXiv][27])
* ELDER — lifelong editing with mixture-of-LoRA and smooth routing. ([arXiv][28])
* LD-MoLE — learnable dynamic routing over LoRA experts. ([arXiv][29])
* MoLoRA — per-token adapter routing. ([arXiv][30])

### Runtime systems

* LoRA-Switch — efficient dynamic adapter switching. ([arXiv][8])
* Activated LoRA — KV-cache reuse across base/adapted models. ([arXiv][31])

### Representation grafts

* ReFT / LoReFT — hidden-state interventions instead of weight deltas. ([arXiv][9])
* CS-ReFT — routed/compositional representation subspaces. ([arXiv][32])

### Hypernetwork / memory compiler direction

* SHINE — context-to-LoRA hypernetwork in one pass. ([arXiv][10])
* Zhyper — factorized hypernetworks for context-aware LoRA. ([arXiv][33])

### Sparse and neural memory substrates

* Sparse Memory Finetuning — sparse memory slots with less forgetting than LoRA. ([arXiv][11])
* Titans — neural long-term memory plus attention as short-term memory. ([arXiv][12])
* ATLAS — high-capacity test-time memory module. ([arXiv][34])
* TNT — hierarchical training for test-time memorization. ([arXiv][35])

### Memory-as-model / agent memory

* MeMo — dedicated memory model, base LLM unchanged. ([arXiv][13])
* Mem-π — generated memory guidance on demand. ([arXiv][14])
* MemoBrain — executive memory for long-horizon tool reasoning. ([arXiv][36])
* MemoryAgentBench — incremental multi-turn benchmark for memory agents. ([arXiv][15])
* MemGym — agentic memory benchmark across tool-use, research, coding, and computer-use regimes. ([arXiv][16])
* Memory in the LLM Era — 2026 survey/unified framework for agent memory. ([arXiv][37])
* H-MEM — hierarchical memory for LLM agents. ([arXiv][17])

### Neuroscience / continual learning inspiration

* Benna & Fusi — multi-timescale biological memory. ([arXiv][18])
* Complementary Learning Systems — fast hippocampal-like and slow cortical-like systems. ([arXiv][19])
* DualNets / CLS-inspired continual learning — fast/slow systems in ML. ([arXiv][38])
* CLS-ER — complementary learning systems with replay. ([arXiv][39])
* Sharp-wave ripples / replay — biological consolidation analogy. ([Wikipedia][40])
* Predictive coding — memory as part of predictive world modeling. ([Wikipedia][41])

---

I’d make **Sparse Memory Finetuning**, **Understanding LoRA as Knowledge Memory**, **MeMo**, **MemGym**, **MemoryAgentBench**, **LD-MoLE/MoLoRA**, and **ReFT/CS-ReFT** the new center of gravity. The older RAG/LoRA/model-editing papers should stay, but mostly as the historical spine rather than the main novelty.

[1]: https://arxiv.org/abs/2005.11401?utm_source=chatgpt.com "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
[2]: https://arxiv.org/abs/2106.09685?utm_source=chatgpt.com "LoRA: Low-Rank Adaptation of Large Language Models"
[3]: https://arxiv.org/abs/2603.01097?utm_source=chatgpt.com "Understanding LoRA as Knowledge Memory: An Empirical Analysis"
[4]: https://arxiv.org/abs/2307.13269?utm_source=chatgpt.com "LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition"
[5]: https://arxiv.org/abs/2410.04891?utm_source=chatgpt.com "Low-Rank Continual Personalization of Diffusion Models"
[6]: https://arxiv.org/abs/2411.00623?utm_source=chatgpt.com "Dual Low-Rank Adaptation for Continual Learning with Pre-Trained Models"
[7]: https://arxiv.org/abs/2312.09979?utm_source=chatgpt.com "LoRAMoE: Alleviate World Knowledge Forgetting in Large Language Models via MoE-Style Plugin"
[8]: https://arxiv.org/abs/2405.17741?utm_source=chatgpt.com "LoRA-Switch: Boosting the Efficiency of Dynamic LLM Adapters via System-Algorithm Co-design"
[9]: https://arxiv.org/abs/2404.03592?utm_source=chatgpt.com "ReFT: Representation Finetuning for Language Models"
[10]: https://arxiv.org/abs/2602.06358?utm_source=chatgpt.com "SHINE: A Scalable In-Context Hypernetwork for Mapping Context to LoRA in a Single Pass"
[11]: https://arxiv.org/abs/2510.15103?utm_source=chatgpt.com "Continual Learning via Sparse Memory Finetuning"
[12]: https://arxiv.org/abs/2501.00663?utm_source=chatgpt.com "Titans: Learning to Memorize at Test Time"
[13]: https://arxiv.org/abs/2605.15156?utm_source=chatgpt.com "MeMo: Memory as a Model"
[14]: https://arxiv.org/abs/2605.21463?utm_source=chatgpt.com "Mem-$π$: Adaptive Memory through Learning When and What to Generate"
[15]: https://arxiv.org/abs/2507.05257?utm_source=chatgpt.com "Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions"
[16]: https://arxiv.org/abs/2605.20833?utm_source=chatgpt.com "MemGym: a Long-Horizon Memory Environment for LLM Agents"
[17]: https://arxiv.org/abs/2507.22925?utm_source=chatgpt.com "Hierarchical Memory for High-Efficiency Long-Term Reasoning in LLM Agents"
[18]: https://arxiv.org/abs/1507.07580?utm_source=chatgpt.com "Computational principles of biological memory"
[19]: https://arxiv.org/abs/1905.02636?utm_source=chatgpt.com "A Complementary Learning Systems Approach to Temporal Difference Learning"
[20]: https://arxiv.org/abs/2002.08909?utm_source=chatgpt.com "REALM: Retrieval-Augmented Language Model Pre-Training"
[21]: https://arxiv.org/abs/2112.04426?utm_source=chatgpt.com "Improving language models by retrieving from trillions of tokens"
[22]: https://arxiv.org/abs/2305.14314?utm_source=chatgpt.com "QLoRA: Efficient Finetuning of Quantized LLMs"
[23]: https://arxiv.org/abs/2210.17451?utm_source=chatgpt.com "AdaMix: Mixture-of-Adaptations for Parameter-efficient Model Tuning"
[24]: https://arxiv.org/abs/2605.01429?utm_source=chatgpt.com "SCALE-LoRA: Auditing Post-Retrieval LoRA Composition with Residual Merging and View Reliability"
[25]: https://arxiv.org/abs/2508.03999?utm_source=chatgpt.com "Tensorized Clustered LoRA Merging for Multi-Task Interference"
[26]: https://arxiv.org/abs/2512.23017?utm_source=chatgpt.com "Merge before Forget: A Single LoRA Continual Learning via Continual Merging"
[27]: https://arxiv.org/abs/2404.15159?utm_source=chatgpt.com "MixLoRA: Enhancing Large Language Models Fine-Tuning with LoRA-based Mixture of Experts"
[28]: https://arxiv.org/abs/2408.11869?utm_source=chatgpt.com "ELDER: Enhancing Lifelong Model Editing with Mixture-of-LoRA"
[29]: https://arxiv.org/abs/2509.25684?utm_source=chatgpt.com "LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts"
[30]: https://arxiv.org/abs/2603.15965?utm_source=chatgpt.com "MoLoRA: Composable Specialization via Per-Token Adapter Routing"
[31]: https://arxiv.org/abs/2512.17910?utm_source=chatgpt.com "Efficient Multi-Adapter LLM Serving via Cross-Model KV-Cache Reuse with Activated LoRA"
[32]: https://arxiv.org/abs/2503.10617?utm_source=chatgpt.com "Compositional Subspace Representation Fine-tuning for Adaptive Large Language Models"
[33]: https://arxiv.org/abs/2510.19733?utm_source=chatgpt.com "Zhyper: Factorized Hypernetworks for Conditioned LLM Fine-Tuning"
[34]: https://arxiv.org/abs/2505.23735?utm_source=chatgpt.com "ATLAS: Learning to Optimally Memorize the Context at Test Time"
[35]: https://arxiv.org/abs/2511.07343?utm_source=chatgpt.com "TNT: Improving Chunkwise Training for Test-Time Memorization"
[36]: https://arxiv.org/abs/2601.08079?utm_source=chatgpt.com "MemoBrain: Executive Memory as an Agentic Brain for Reasoning"
[37]: https://arxiv.org/abs/2604.01707?utm_source=chatgpt.com "Memory in the LLM Era: Modular Architectures and Strategies in a Unified Framework"
[38]: https://arxiv.org/abs/2209.02370?utm_source=chatgpt.com "Continual Learning, Fast and Slow"
[39]: https://arxiv.org/abs/2201.12604?utm_source=chatgpt.com "Learning Fast, Learning Slow: A General Continual Learning Method based on Complementary Learning System"
[40]: https://en.wikipedia.org/wiki/Sharp_waves_and_ripples?utm_source=chatgpt.com "Sharp waves and ripples"
[41]: https://en.wikipedia.org/wiki/Predictive_coding?utm_source=chatgpt.com "Predictive coding"
