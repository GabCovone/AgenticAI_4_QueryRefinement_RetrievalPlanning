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
    Applies Phase 1 of the 'Least-to-Most Prompting' framework.
    WHEN TO USE: Use this tool when the query is complex (multi-hop), contains multiple main subjects, or requires solving intermediate problems.
    INSTRUCTIONS: In the 'reflection' field, critique your past attempts based on the Validator's feedback. In 'reasoning', explain why you chose to decompose.
    CRITICAL RULE: A multi-hop query must be broken down into a SEQUENCE of dependent steps. Step 1 should ask to identify the unknown entity (e.g., "Who is the director of the film Inception?"). Step 2 should ask the final question using a placeholder for the result of Step 1 (e.g., "In which city was [Director of Inception] born?"). DO NOT create two sub-queries that ask the exact same thing.
    CRITICAL RULE 2: Do not resolve unknown entities. BUT ALWAYS KEEP EXPLICIT ENTITIES (e.g. KEEP "Inception", do not replace it with "[Film Name]").
    EXAMPLE: If the query is "In which city was the director of the film Inception born?"
    The `sub_queries` argument must be: ["Who is the director of the film Inception?", "In which city was [Director of Inception] born?"]
    """
    return sub_queries

@tool
def expand_query(reflection: str, reasoning: str, pseudo_document: str) -> str:
    """
    Applies the 'Query2doc' framework for semantic expansion.
    WHEN TO USE: Use this tool when the query is clear and has a single subject, but is too short, vague, or lacks the technical jargon needed for effective retrieval.
    INSTRUCTIONS: In the 'reflection' field, critique your past attempts based on the Validator's feedback. In 'reasoning', explain why you are expanding it.
    CRITICAL RULE: Do not resolve unknown entities. Use placeholders for unknown entities. BUT ALWAYS KEEP EXPLICIT ENTITIES.
    EXAMPLE: If the query is "car batteries that don't die in the cold".
    The `pseudo_document` argument must be: "Solid-state batteries, lithium iron phosphate (LFP), thermal degradation, low-temperature efficiency, EV winter range, pre-conditioning."
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
    WHEN TO USE: If the query contains a placeholder (like [Director of Inception]), or if the Validator feedback indicates it is ready for retrieval (e.g. 'Proceed with document retrieval').
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
1. AUTONOMY: The Validator's feedback is just a suggestion. If the query is ALREADY a single, atomic step (e.g., containing a single placeholder like `[Director of Inception]`), IGNORE the Validator's suggestion to decompose and call `keep_query_unchanged`.
2. ENTITIES: DO NOT resolve unknown entities using internal knowledge. Use placeholders (e.g., "[Director of X]"). ALWAYS KEEP explicit entities mentioned by the user (e.g., keep "Inception", do not replace with "[Film]").
3. LANGUAGE: Use strictly English. Do not output Chinese characters.

TOOL SELECTION GUIDELINES:
- 'decompose_query': If the query is multi-hop, contains multiple concepts, or requires intermediate steps.
- 'keep_query_unchanged': If the query is already a simple, single, atomic question (even with placeholders) or ready for retrieval.
- 'rewrite_query': To remove conversational noise.
- 'expand_query': To add context if previous retrieval failed.

You MUST formulate your Chain-of-Thought inside the `reasoning` field and critique feedback in the `reflection` field.
--- FEW-SHOT EXAMPLES ---

User's query: "In which city was the director of the film Inception born?"
Expected Action: The query is multi-hop. We need to identify the director first, and then their birthplace.
Tools to call:
{
  "name": "decompose_query",
  "arguments": {
    "reflection": "The Validator feedback indicates that the query is multi-hop and requires intermediate steps.",
    "reasoning": "I must split this into a sequence. First, identify the director of Inception. Second, ask for the birthplace of that director using a placeholder.",
    "sub_queries": ["Who is the director of the film Inception?", "In which city was [Director of Inception] born?"]
  }
}

User's query: "car batteries that don't die in the cold"
Expected Action: The query needs technical jargon.
Tools to call:
{
  "name": "expand_query",
  "arguments": {
    "reflection": "The query lacks technical terms for effective retrieval.",
    "reasoning": "Adding technical terminology about solid-state batteries and cold weather performance will improve search.",
    "pseudo_document": "Solid-state batteries, lithium iron phosphate (LFP), thermal degradation, low-temperature efficiency, EV winter range, pre-conditioning."
  }
}

User's query: "who is the mother of the guy who wrote the book harry potter?"
Expected Action: The query is conversational and contains an unknown entity (the author).
Tools to call:
{
  "name": "rewrite_query",
  "arguments": {
    "reflection": "The query contains conversational noise and asks to resolve an entity ('the guy who wrote').",
    "reasoning": "I need to rewrite this formally and USE A PLACEHOLDER for the unknown author, without resolving it to J.K. Rowling.",
    "rewritten_query": "Identify the mother of [Author of Harry Potter]"
  }
}

User's query: "In which city was [Director of Inception] born?"
Expected Action: The query is already atomic and contains a placeholder.
Tools to call:
{
  "name": "keep_query_unchanged",
  "arguments": {
    "reflection": "The Validator feedback indicates it is ready for retrieval, and the query is a single atomic question with a placeholder.",
    "reasoning": "Since it is already an atomic step, I will leave it unchanged."
  }
}
"""

    feedback_history = state.get("feedback_history", [])
    feedback_text = ""
    if feedback_history:
        feedback_text = f"\n\nFEEDBACK FROM VALIDATOR (CRITICAL: You must address these instructions):\n- " + "\n- ".join(feedback_history)

    final_processed_queries = []
    tools_used_global = set()

    for q_idx, q in enumerate(queries):
        print(f"[REFINER] Analisi sotto-query {q_idx+1}/{len(queries)}: '{q}'")
        try:
            response_msg = llm_with_tools.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Sub-query to analyze: '{q}'. Analyze the information need and call the appropriate tools.{feedback_text}")
            ])
        except Exception as e:
            print(f"[REFINER ERROR] Inferenza fallita per la sub-query: {e}")
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
                print(f"[REFINER DEBUG] Tool Call '{tool_call['name']}' con args: {args}")
                
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

            # Priorità: DECOMPOSE > REWRITE > EXPAND > KEEP per questa specifica sub-query
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
