from typing import List
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from graph import GraphState

# --- 1. DEFINIZIONE DEI TOOL CON DESCRIZIONI SEMANTICHE (DOCSTRINGS) ---
# Le docstring vengono lette dal modello e formano lo schema del tool.
# Devono spiegare in modo esaustivo l'intento logico, i limiti e fornire esempi specifici.

@tool
def decompose_query(reflection: str, reasoning: str, sub_queries: List[str]) -> List[str]:
    """
    Applies parallel decomposition.
    WHEN TO USE: Use this tool ONLY when the query contains MULTIPLE INDEPENDENT entities or facts that must be searched separately (e.g., comparisons, intersections). DO NOT use this for multi-hop sequential queries (e.g., "Where was the director of X born?"), as the Planner can handle those.
    INSTRUCTIONS: In the 'reflection' field, critique your past attempts based on the Validator's feedback. In 'reasoning', explain why you chose to decompose.
    CRITICAL RULE: Break the query down into entirely independent sub-queries that can be executed in parallel.
    CRITICAL RULE 2: Do not resolve unknown entities. BUT ALWAYS KEEP EXPLICIT ENTITIES.
    EXAMPLE: If the query is "Were Scott Derrickson and Ed Wood of the same nationality?"
    The `sub_queries` argument must be: ["What is the nationality of Scott Derrickson?", "What is the nationality of Ed Wood?"]
    """
    return sub_queries

@tool
def expand_query(reflection: str, reasoning: str, pseudo_document: str) -> str:
    """
    Applies the 'Query2doc' framework for semantic expansion.
    WHEN TO USE: Use this tool when the query is short, ambiguous, or lacks necessary background information.
    INSTRUCTIONS: Generate a 'pseudo-document' that attempts to answer the query using your internal knowledge. This generated passage will be concatenated with the original query to guide the retrieval system by providing rich contextual vocabulary.
    CRITICAL RULE: Write a factual passage that answers the query or provides highly relevant background details.
    EXAMPLE: If the query is "when was pokemon green released"
    The `pseudo_document` argument must be: "Pokemon Green was released in Japan on February 27th, 1996. It was the first in the Pokemon series of games and served as the basis for Pokemon Red and Blue."
    """
    return pseudo_document

@tool
def rewrite_query(reflection: str, reasoning: str, rewritten_query: str) -> str:
    """
    Applies the 'Rewrite-Retrieve-Read' framework for syntactic correction.
    WHEN TO USE: Use this tool for single queries that contain conversational noise (e.g., "hey can you tell me..."), grammatical errors, or convoluted syntax.
    INSTRUCTIONS: In the 'reflection' field, critique your past attempts based on the Validator's feedback. In 'reasoning', explain your changes.
    CRITICAL RULE: Do not resolve unknown entities. Use placeholders for unknown entities (e.g. [Director of X]). BUT ALWAYS KEEP EXPLICIT ENTITIES (e.g. KEEP "Inception", do not replace it with "[Film Name]").
    EXAMPLE: If the query is "hey listen but how do you figure out if the car engine has a broken head gasket?".
    The `rewritten_query` argument must be: "Symptoms and diagnostic methods for a damaged engine head gasket".
    """
    return rewritten_query


@tool
def keep_query_unchanged(reflection: str, reasoning: str) -> str:
    """
    Use this tool when the sub-query is already a simple, single, atomic question ready for retrieval.
    WHEN TO USE: ONLY if the query asks ONE SINGLE thing about ONE SINGLE entity. NEVER use this if the query contains 'and', 'or', 'both', or compares two entities (you must use decompose_query instead).
    INSTRUCTIONS: In the 'reflection' field, acknowledge the Validator's feedback and explicitly state that no refinement is needed.
    """
    return "UNCHANGED"

# --- 2. NODO REFINER ---
def refiner_node(state: GraphState, llm) -> dict:
    print("\n--- [REFINER] Inizio analisi della query ---")
    current_query_raw = state.get("current_query", state.get("original_query", ""))
    
    # Dividiamo le sub-query per analizzarle singolarmente!
    queries = [q.strip() for q in current_query_raw.split("\n---\n") if q.strip()]
    if not queries:
        queries = [current_query_raw]
        
    tools = [decompose_query, expand_query, rewrite_query, keep_query_unchanged]
    llm_with_tools = llm.bind_tools(tools)
    
    system_prompt = """You are the Autonomous Query Refinement Agent.
Optimize the given query or sub-query using the provided tools.

CRITICAL RULES:
1. AUTONOMY: The Validator's feedback is just a suggestion. If the query contains MULTIPLE distinct entities being compared or intersected (e.g., 'X and Y', 'both A and B'), it is NOT atomic and MUST be decomposed. However, if the query has only ONE main entity or is a multi-hop sequential question (e.g., 'Who is the director of X?'), it IS atomic for your purposes, so you MUST ignore the Validator's suggestion to decompose and call `keep_query_unchanged`.
2. ENTITIES: DO NOT resolve unknown entities using internal knowledge. Use placeholders for UNKNOWN entities (e.g., "[Director of X]"). NEVER wrap explicit, known names in placeholders (e.g., if the user asks about "Scott Derrickson", DO NOT output "[Director Scott Derrickson]". Keep it as "Scott Derrickson").
3. LANGUAGE: Use strictly English. Do not output Chinese characters.

TOOL SELECTION GUIDELINES:
- 'decompose_query': ONLY if the query consists of entirely independent parallel questions (e.g., "Are X and Y the same nationality?"). DO NOT use for multi-hop sequential questions.
- 'keep_query_unchanged': If the query is multi-hop (Planner will handle it via ReAct), or already a simple atomic question, or ready for retrieval.
- 'rewrite_query': To remove conversational noise.
- 'expand_query': To generate a Query2doc pseudo-document and add semantic context.

CRITICAL INSTRUCTION: You do not support native function calling. You MUST manually output a RAW JSON object representing the tool call. DO NOT output any conversational text.

You MUST formulate your Chain-of-Thought inside the `reasoning` field and critique feedback in the `reflection` field.
--- FEW-SHOT EXAMPLES ---

User's query: "Were Scott Derrickson and Ed Wood of the same nationality?"
Expected Action: The query requires comparing two independent entities.
Tools to call:
```json
{
  "name": "decompose_query",
  "arguments": {
    "reflection": "The Validator feedback indicates parallel searches are needed.",
    "reasoning": "I must split this into independent queries to find the nationality of each person separately before comparing.",
    "sub_queries": ["What is the nationality of Scott Derrickson?", "What is the nationality of Ed Wood?"]
  }
}
```

User's query: "who is the mother of the guy who wrote the book harry potter?"
Expected Action: The query contains an unknown entity (the author).
Tools to call:
```json
{
  "name": "rewrite_query",
  "arguments": {
    "reflection": "The query asks to resolve an entity ('the guy who wrote').",
    "reasoning": "I need to rewrite this formally and USE A PLACEHOLDER for the unknown author, without resolving it.",
    "rewritten_query": "Identify the mother of [Author of Harry Potter]"
  }
}
```

User's query: "when was pokemon green released"
Expected Action: The query is unambiguous but needs background knowledge expansion.
Tools to call:
```json
{
  "name": "expand_query",
  "arguments": {
    "reflection": "The Validator feedback suggests search expansion due to lack of specific context.",
    "reasoning": "I need to generate a pseudo-document from my internal knowledge to enrich the retrieval system's vocabulary.",
    "pseudo_document": "Pokemon Green was released in Japan on February 27th, 1996. It was the first in the Pokemon series of games and served as the basis for Pokemon Red and Blue."
  }
}
```
"""

    feedback_history = state.get("feedback_history", [])
    feedback_text = ""
    if feedback_history:
        feedback_text = f"\n\nFEEDBACK FROM VALIDATOR (CRITICAL: You must address these instructions):\n- " + "\n- ".join(feedback_history)

    final_processed_queries = []
    tools_used_global = set()

    for q_idx, q in enumerate(queries):
        print(f"\n🔍 [REFINER] Analisi sotto-query {q_idx+1}/{len(queries)}: '{q}'")
        try:
            full_prompt = f"{system_prompt}\n\nOriginal Query: '{current_query_raw}'\nCurrent Sub-Query: '{q}'\n\nFeedback from Validator:{feedback_text}\n\nAnalyze the feedback and output the JSON tool call."
            response_msg = llm_with_tools.invoke([
                HumanMessage(content=full_prompt)
            ])
        except Exception as e:
            print(f"❌ [REFINER ERROR] Inferenza fallita per la sub-query: {e}")
            final_processed_queries.append(q)
            continue
            
        tool_calls_list = []
        # Use native tool_calls only if they contain actual arguments
        if hasattr(response_msg, "tool_calls") and response_msg.tool_calls and all(tc.get("args") for tc in response_msg.tool_calls):
            tool_calls_list = response_msg.tool_calls
        else:
            import json
            content_resp = getattr(response_msg, "content", str(response_msg))
            i = 0
            while i < len(content_resp):
                if content_resp[i] == '{':
                    brace_level = 0
                    in_string = False
                    escape = False
                    for j in range(i, len(content_resp)):
                        char = content_resp[j]
                        if char == '"' and not escape:
                            in_string = not in_string
                        if not in_string:
                            if char == '{': brace_level += 1
                            elif char == '}': brace_level -= 1
                        if brace_level == 0:
                            try:
                                parsed = json.loads(content_resp[i:j+1])
                                if isinstance(parsed, dict):
                                    t_name = None
                                    t_args = {}
                                    if "name" in parsed:
                                        t_name = parsed["name"]
                                        t_args = parsed.get("arguments", parsed.get("args", parsed.get("parameters", parsed)))
                                    else:
                                        # Infer name from keys if 'name' wrapper is missing
                                        t_args = parsed
                                        if "sub_queries" in t_args or "queries" in t_args:
                                            t_name = "decompose_query"
                                        elif "rewritten_query" in t_args or "query" in t_args:
                                            t_name = "rewrite_query"
                                        elif "pseudo_document" in t_args or "expansion" in t_args:
                                            t_name = "expand_query"
                                        else:
                                            t_name = "keep_query_unchanged"
                                            
                                    if t_name:
                                        tool_calls_list.append({"name": t_name, "args": t_args})
                            except Exception: pass
                            i = j
                            break
                        if char == '\\': escape = not escape
                        else: escape = False
                i += 1

        if tool_calls_list:
            decomposed_queries = None
            rewritten_query = None
            expansion_text = None
            keep_unchanged = False
            for tool_call in tool_calls_list:
                tools_used_global.add(tool_call['name'])
                args = tool_call['args']
                print(f"   💭 [REFINER] Reasoning: {args.get('reasoning', 'No reasoning provided')}")
                print(f"   🛠️  [REFINER] Tool Call '{tool_call['name']}' con args: {args}")
                
                if tool_call['name'] == "decompose_query":
                    decomposed_queries = args.get("sub_queries", args.get("queries"))
                    if isinstance(decomposed_queries, str):
                        try:
                            import json
                            decomposed_queries = json.loads(decomposed_queries)
                        except Exception:
                            import ast
                            try:
                                decomposed_queries = ast.literal_eval(decomposed_queries)
                            except Exception:
                                decomposed_queries = [decomposed_queries]
                elif tool_call['name'] == "rewrite_query":
                    rewritten_query = args.get("rewritten_query", args.get("query"))
                elif tool_call['name'] == "expand_query":
                    expansion_text = args.get("pseudo_document", args.get("expansion", args.get("context", args.get("expanded_context"))))
                elif tool_call['name'] == "keep_query_unchanged":
                    keep_unchanged = True

            # Priorità: DECOMPOSE > REWRITE > EXPAND > KEEP
            if decomposed_queries:
                final_processed_queries.extend(decomposed_queries)
            elif rewritten_query:
                final_processed_queries.append(rewritten_query)
            elif expansion_text:
                final_processed_queries.append(f"{q} \n[Expanded Context]: {expansion_text}")
            elif keep_unchanged:
                final_processed_queries.append(q)
            else:
                final_processed_queries.append(q)
        else:
            final_processed_queries.append(q)

    # Ricomponiamo lo stato
    final_query_str = "\n---\n".join(final_processed_queries)
    if tools_used_global:
        print(f"[REFINER] Tool globali applicati: {', '.join(tools_used_global).upper()}")
    else:
        print("[REFINER] L'agente ha ritenuto tutte le query perfette. Nessun tool chiamato.")
    print(f"[REFINER] Output finale:\n{final_query_str}")
    
    return {
        "current_query": final_query_str,
        "num_refinement": state.get("num_refinement", 0) + 1
    }
