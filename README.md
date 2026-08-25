# Agentic Information Retrieval (IR) System

This project implements an Agentic Information Retrieval (IR) system based on Large Language Models (LLMs). 
The goal is to overcome the limitations of traditional unidirectional Retrieval-Augmented Generation (RAG) systems by introducing a cyclical and self-reflexive multi-agent architecture. 

The system is able to manage complex information needs, assess the ambiguity of user queries, decompose them, plan multi-step retrieval strategies, and self-correct in the event of missing or contradictory information. Flow orchestration is performed using **LangGraph**, while the agentic logic is entrusted to a **Qwen 14B Instruct** model (quantized for execution on resource-constrained hardware such as Google Colab).

## System Architecture

The system is based on a global state graph (`GraphState`) shared between three main modules. To optimize performance and reduce computational costs (and network cycles on the main graph), the architecture delegates "micro-reflection" within individual agents, which operate partially autonomously before returning control to the central router. 

The flow follows this life cycle: 
1. **Input**: The user enters the initial query. 
2. **Routing**: The Validator analyzes the query and decides whether to route it to the Refiner or the Planner. 
3. **Autonomous Processing**: The designated module executes its task (with possible internal self-criticism loops). 
4. **Final Evaluation**: Execution returns to the Validator, which judges the results. If the output is satisfactory, the cycle ends; otherwise, it generates textual feedback and triggers a new refinement or planning cycle. 

## Modules and Academic Foundations 

Each module of the system has been designed by implementing specific state-of-the-art (SOTA) theoretical frameworks in the fields of Agentic AI and Natural Language Processing. 

### 1. The Validator (`Validator.py`)
Acts as the Conductor (input router) and Judge (output). It queries the model by imposing a rigorous JSON output schema (`ValidatorDecision`). 
* **Sliding-Window Try-Parse**: Features a custom, highly robust JSON fallback parser that scans the output independently evaluating every `{`. It provides total immunity to "Ghost Token Hallucinations" and premature JSON block restarts, seamlessly intercepting valid JSONs even when hidden behind corrupted formatting.
* **Cross-Lingual Guardrails**: System prompts, few-shot examples, and external API calls (e.g. DuckDuckGo region locking) are strictly enforced in English to prevent cross-lingual cognitive load and "Few-Shot Leakage" (lazy copying of examples by the LLM during context confusion).
* **Adaptive-RAG**: The model is forced to explicitly assess the input via a `query_complexity` attribute (simple, complex, already_decomposed), which mathematically dictates the initial routing strategy (Jeong et al., 2024). 
* **Self-RAG**: After searches are executed, the Validator generates two explicit *Critique Tokens* (`is_context_relevant` and `is_query_answered`) to evaluate whether the extracted documents are semantically relevant and sufficient to logically support the original query (Asai et al., 2023). 
* **Reflexion**: If the task fails, the Validator leverages a dedicated `reflection` attribute to critique the trajectory (e.g. Planning Count loops) and outputs a textual instruction for the next loop to avoid repeating mistakes (Shinn et al., 2023). 

### 2. The Refiner (`Refiner.py`)
Module responsible for **Autonomous Query Refinement**. It does not perform database searches, but prepares the perfect query via tool-calling. Depending on the problem detected in the input query, the model autonomously chooses which tool to apply (even in parallel). To enforce the **Reflexion** framework, all tools require a mandatory `reflection` parameter where the LLM must critique its past attempts based on the Validator's feedback before acting.
* **Decomposition Tool**: Implements Least-to-Most Prompting (Zhou et al., 2022) to divide complex queries into logical and sequential sub-problems. 
* **Rewriting Tool**: Implements the Rewrite-Retrieve-Read framework (Ma et al., 2023) to correct ambiguous or poorly formulated queries. 
* **Semantic Expansion Tool**: Exploits the Query2doc paradigm (Wang et al., 2023) to generate pseudo-documents or lists of key concepts to be concatenated to the original query, in order to maximize recall in the next vector step.

### 3. The Planner (`Planner.py`)
The operational heart of the system, responsible for **Multi-Step Retrieval Planning**. It receives queries from the Refiner, classifies the required strategy, and actively accesses external search tools (vector databases, open web via DuckDuckGo). 
* **Strategic Classification**: It uses Adaptive-RAG (Jeong et al., 2024) again to decide whether the sub-query requires internal knowledge, a single retrieval step, or a multi-step plan. 
* **Context-Aware Execution (IRCoT)**: During `single_retrieval` and `internal_knowledge`, it explicitly injects the `global_context` of previously answered sub-queries to dynamically resolve missing entities and placeholders (e.g. replacing pronouns with discovered names) before querying the search engine.
* **Multi-Step ReAct**: Orchestrates data retrieval by intertwining reasoning and action based on the ReAct operation loop (Yao et al., 2022), formulating dynamic follow-ups (Self-Ask) if the retrieved context is missing crucial information.

## Technology Stack and Setup
* **Orchestration**: `langgraph` (state management and conditional loops). 
* **LLM Interaction**: `langchain` (tool binding and structured output). 
* **Model**: `Qwen3-14B-Instruct` (GGUF Quantization Q4_K_M). 
* **Inference Engine**: `llama-cpp-python` (compiled with CUDA 12.4+ support for full GPU offload). 

> **Hardware note**: The 14 billion parameter, 4-bit quantized model was selected to fit within the VRAM (16 GB) limits of the NVIDIA T4 GPU provided by Google's free Colab tier, while providing superior logic and JSON formatting capabilities compared to standard 7B models.

## References 
The system is based on the following academic studies (sorted by role in the project): 

**Core Literature:**
1. Zhang, Y. et al. (2024). *AI Agent for Information Retrieval: Generating and Ranking*. CIKM '24 Workshop. 
2. Wu, Y. et al. (2025). *Agentic Reasoning: Reasoning LLMs with Tools for the Deep Research*. 
3. Lála, J. et al. (2023). *PaperQA: Retrieval-Augmented Generative Agent for Scientific Research*. 

**Routing, Reflection, and Evaluation (Validator):**
4. Asai, A. et al. (2023). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*. 
5. Shinn, N. et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. 

**Query Refinement and Expansion (Refiner):**
6. Zhou, D. et al. (2022). *Least-to-Most Prompting Enables Complex Reasoning in Large Language Models*.
7. Ma, X. et al. (2023). *Query Rewriting for Retrieval-Augmented Large Language Models*.
8. Wang, L. et al. (2023). *Query2doc: Query Expansion with Large Language Models*.

**Multi-Step Classification and Planning (Planner):**
9. Jeong, S. et al. (2024). *Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity*.
10. Yao, S. et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*.
11. Trivedi, H. et al. (2022). *Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions (IRCoT)*.
12. Press, O. et al. (2022). *Measuring and Narrowing the Compositionality Gap in Language Models (Self-Ask)*.