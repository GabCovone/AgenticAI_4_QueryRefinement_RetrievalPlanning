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
    CRITICAL RULE: DO NOT answer the questions or resolve entities using your internal knowledge! You MUST use placeholders (like [Person], [Movie Name]) for entities that need to be discovered in previous steps.
    EXAMPLE: If the query is "Who wrote the soundtrack for the movie that won the Oscar in 2020?"
    The `sub_queries` argument must be: ["Which movie won the Oscar for best picture in 2020?", "Who composed the soundtrack for the movie [Movie Name]?"] (DO NOT write 'Parasite', use the placeholder).
    """
    return sub_queries

@tool
def expand_query(reflection: str, reasoning: str, pseudo_document: str) -> str:
    """
    Applies the 'Query2doc' framework for semantic expansion.
    WHEN TO USE: Use this tool when the query is clear and has a single subject, but is too short, vague, or lacks the technical jargon needed for effective retrieval.
    INSTRUCTIONS: In the 'reflection' field, critique your past attempts based on the Validator's feedback. In 'reasoning', explain why you are expanding it.
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
    EXAMPLE: If the query is "hey listen but how do you figure out if the car engine has a broken head gasket?".
    The `rewritten_query` argument must be: "Symptoms and diagnostic methods for a damaged engine head gasket".
    """
    return rewritten_query


# --- 2. NODO REFINER ---
def refiner_node(state: GraphState, llm) -> dict:
    """
    Autonomous Query Refinement.
    Usa il Parallel Tool Calling interpretando logicamente le docstring dei tool.
    """
    print("\n--- [REFINER] Inizio analisi della query ---")
    current_query = state.get("current_query", state.get("original_query", ""))
    
    # Bind dei tool al modello (ora le docstring complete vengono passate a Qwen)
    tools = [decompose_query, expand_query, rewrite_query]
    llm_with_tools = llm.bind_tools(tools)
    
    # --- 3. SYSTEM PROMPT CON CHAIN-OF-THOUGHT FEW-SHOTS ---
    system_prompt = """You are the Autonomous Query Refinement Agent.
Your goal is to analyze the user's query and decide how to optimize it using your tools.
You can use multiple tools simultaneously.
Since you must return formatted outputs, you **MUST** formulate your Chain-of-Thought inside the `reasoning` field of the tools you call.
Additionally, you **MUST** use the `reflection` field to explicitly analyze any feedback provided by the Validator and critique your past attempts before making a decision.

CRITICAL LANGUAGE RULE: YOU MUST ONLY USE ENGLISH. Do not use or output any Chinese characters under any circumstances.
CRITICAL RULE FOR ALL TOOLS: DO NOT answer the questions or resolve unknown entities using your internal knowledge! If the query asks for "the author of X" or "the mother of Y", YOU MUST NOT write their actual names in your refined queries or expanded contexts. You MUST use generic placeholders (like [Author's Name], [Person's Mother]).

TOOL SELECTION GUIDELINES:
1. If the query contains multiple concepts or distinct questions -> Call `decompose_query`.
   - CRITICAL RULE: If the user's query ALREADY contains "---", it has ALREADY been decomposed. DO NOT call `decompose_query` or `rewrite_query` again! You MUST ONLY call `expand_query` to append context.
2. If the query is conversational, noisy, or grammatically incorrect -> Call `rewrite_query`.
3. If the query is too short or lacks technical terminology -> Call `expand_query`.

PIPELINE CASCADE EXECUTION:
You can combine tools. If you call multiple tools, they form a cascade pipeline:
- Decomposition OR Rewriting happens first to establish the base queries.
- Then, Semantic Expansion is generated and appended to the result.

--- FEW-SHOT EXAMPLES ---

User's query: "climate effects on gdp and laws to mitigate it"
Expected Action: The query contains two distinct topics.
Tools to call:
{
  "name": "decompose_query",
  "arguments": {
    "reflection": "The Validator feedback (if any) indicates that the query is too complex for a single search. I must split it.",
    "reasoning": "The user asks about economic impact (GDP) and legal regulations (laws). A single vector search will fail, so I must split them.",
    "sub_queries": ["What are the economic effects of climate change on GDP?", "What laws and regulations are implemented to mitigate climate change?"]
  }
}

User's query: "what state is this zip code 85282"
Expected Action: Single factual question, but too short for semantic search.
Tools to call:
{
  "name": "expand_query",
  "arguments": {
    "reflection": "There is no critical feedback, but the query is very short. I should expand it to improve recall.",
    "reasoning": "The query is clear but lacks vocabulary. I will generate a pseudo-document with geographical context to improve vector recall.",
    "pseudo_document": "Arizona, Maricopa County, Tempe, postal codes geography, US states, Southwest region."
  }
}

User's query: "hey can u tell me how to fix physics crash returnal"
Expected Action: Conversational noise, poor syntax, lacks technical terms. Needs rewrite AND expansion.
Tools to call (Parallel):
{
  "name": "rewrite_query",
  "arguments": {
    "reflection": "The feedback indicates the query contains too much noise. I need to extract the core informational intent.",
    "reasoning": "Removing conversational noise and formalizing syntax for search.",
    "rewritten_query": "How to resolve physics engine crashes in the game Returnal"
  }
}
{
  "name": "expand_query",
  "arguments": {
    "reflection": "The feedback also suggests adding technical jargon to broaden the search.",
    "reasoning": "Adding technical gaming context to expand the search surface.",
    "pseudo_document": "Unreal Engine 4, PS5, PC port, Housemarque, collision bug, patch update, fatal error, graphics drivers, GPU crash."
  }
}

IMPORTANT: You can think out loud first (keep it extremely brief), but you MUST conclude your response with VALID JSON BLOCKS as shown above.
"""

    try:
        feedback_history = state.get("feedback_history", [])
        feedback_text = ""
        if feedback_history:
            feedback_text = f"\n\nFEEDBACK FROM VALIDATOR (CRITICAL: You must address these instructions):\n- " + "\n- ".join(feedback_history)
            
        # L'LLM processerà il ragionamento e chiamerà i tool appropriati
        response_msg = llm_with_tools.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User's query: '{current_query}'. Analyze the information need and call the appropriate tools.{feedback_text}")
        ])
    except Exception as e:
        print(f"[REFINER ERROR] Inferenza fallita: {e}")
        return {"current_query": current_query} 

    # --- 4. GESTIONE DELLA PIPELINE IN PYTHON ---
    base_queries = [current_query]
    expansion_text = ""
    tools_used = []
    
    decomposed_queries = None
    rewritten_query = None

    if hasattr(response_msg, "tool_calls") and response_msg.tool_calls:
        tool_calls_list = response_msg.tool_calls
    else:
        # FALLBACK JSON PARSER SUPER-ROBUSTO
        import json
        tool_calls_list = []
        content = getattr(response_msg, "content", str(response_msg))
        
        # NUOVO PARSER: Cerca ogni '{' e tenta di estrarre un JSON valido (supporta multiple tool calls)
        i = 0
        while i < len(content):
            if content[i] == '{':
                brace_level = 0
                in_string = False
                escape = False
                for j in range(i, len(content)):
                    char = content[j]
                    if char == '"' and not escape:
                        in_string = not in_string
                    
                    if not in_string:
                        if char == '{':
                            brace_level += 1
                        elif char == '}':
                            brace_level -= 1
                            
                    if brace_level == 0:
                        try:
                            parsed = json.loads(content[i:j+1])
                            if "name" in parsed:
                                t_name = parsed["name"]
                                t_args = parsed.get("arguments", parsed.get("args", {}))
                                tool_calls_list.append({"name": t_name, "args": t_args})
                        except Exception:
                            pass
                        i = j # Salta i caratteri già elaborati
                        break
                        
                    if char == '\\':
                        escape = not escape
                    else:
                        escape = False
            i += 1
                
    if tool_calls_list:
        for tool_call in tool_calls_list:
            tools_used.append(tool_call['name'])
            args = tool_call['args']
            
            # Recupero degli argomenti generati dal tool-calling
            if tool_call['name'] == "decompose_query" and "sub_queries" in args:
                decomposed_queries = args["sub_queries"]
            elif tool_call['name'] == "rewrite_query" and "rewritten_query" in args:
                rewritten_query = args["rewritten_query"]
            elif tool_call['name'] == "expand_query" and "pseudo_document" in args:
                expansion_text = args["pseudo_document"]

        # Logica di risoluzione (priorità a decompose se presenti entrambi)
        if decomposed_queries:
            base_queries = decomposed_queries
        elif rewritten_query:
            base_queries = [rewritten_query]

        # Pipe di concatenazione (Applica Query2doc alle query base/riscritte)
        if expansion_text:
            final_queries = [f"{q} \n[Expanded Context]: {expansion_text}" for q in base_queries]
        else:
            final_queries = base_queries
            
        # Uniamo le query in una singola stringa perché GraphState.current_query è di tipo str
        final_query_str = "\n---\n".join(final_queries)
            
        print(f"[REFINER] Tool applicati: {', '.join(tools_used).upper()}")
        print(f"[REFINER] Output finale:\n{final_query_str}")
        
        return {
            "current_query": final_query_str,
            "num_refinement": state.get("num_refinement", 0) + 1
        }
    else:
        print("[REFINER] L'agente ha ritenuto la query perfetta. Nessun tool chiamato.")
        return {
            "current_query": current_query,
            "num_refinement": state.get("num_refinement", 0) + 1
        }