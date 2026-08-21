from typing import List
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from graph import GraphState

# --- 1. DEFINIZIONE DEI TOOL CON DESCRIZIONI SEMANTICHE (DOCSTRINGS) ---
# Le docstring vengono lette dal modello e formano lo schema del tool.
# Devono spiegare in modo esaustivo l'intento logico, i limiti e fornire esempi specifici.

@tool
def decompose_query(reasoning: str, sub_queries: List[str]) -> List[str]:
    """
    Applies Phase 1 of the 'Least-to-Most Prompting' framework.
    WHEN TO USE: Use this tool when the query is complex (multi-hop), contains multiple main subjects, or requires solving intermediate problems.
    WHY TO USE: Vector databases fail if they try to retrieve multiple distinct concepts at once. 
    INSTRUCTIONS: Analyze the query and decompose it into a list of atomic, logically sequential, and independent sub-queries. In the 'reasoning' field, explain why you chose to decompose.
    EXAMPLE: If the query is "Who wrote the soundtrack for the movie that won the Oscar in 2020?", you cannot solve it in one step.
    The `sub_queries` argument must be: ["Which movie won the Oscar for best picture in 2020?", "Who composed the soundtrack for the movie [Movie Name]?"]
    """
    return sub_queries

@tool
def expand_query(reasoning: str, pseudo_document: str) -> str:
    """
    Applies the 'Query2doc' framework for semantic expansion.
    WHEN TO USE: Use this tool when the query is clear and has a single subject, but is too short, vague, or lacks the technical jargon needed for effective retrieval.
    WHY TO USE: Generating related concepts, synonyms, and a hypothetical context (pseudo-document) drastically improves the Recall of the vector database.
    INSTRUCTIONS: Generate a short text string containing latent keywords, acronyms, or synonyms related to the query. In the 'reasoning' field, explain why you are expanding it.
    EXAMPLE: If the query is "car batteries that don't die in the cold".
    The `pseudo_document` argument must be: "Solid-state batteries, lithium iron phosphate (LFP), thermal degradation, low-temperature efficiency, EV winter range, pre-conditioning."
    """
    return pseudo_document

@tool
def rewrite_query(reasoning: str, rewritten_query: str) -> str:
    """
    Applies the 'Rewrite-Retrieve-Read' framework for syntactic correction.
    WHEN TO USE: Use this tool for single queries that contain conversational noise (e.g., "hey can you tell me..."), grammatical errors, or convoluted syntax.
    WHY TO USE: Removing noise and linearizing the syntax focuses the search engine's attention exclusively on the informational intent.
    INSTRUCTIONS: Rewrite the entire question in a formal, direct manner, optimized for a search engine. In the 'reasoning' field, explain your changes.
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

TOOL SELECTION GUIDELINES:
1. If the query contains multiple concepts or distinct questions -> Call `decompose_query`.
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
- decompose_query
  - reasoning: "The user asks about economic impact (GDP) and legal regulations (laws). A single vector search will fail, so I must split them."
  - sub_queries: ["What are the economic effects of climate change on GDP?", "What laws and regulations are implemented to mitigate climate change?"]

User's query: "what state is this zip code 85282"
Expected Action: Single factual question, but too short for semantic search.
Tools to call:
- expand_query
  - reasoning: "The query is clear but lacks vocabulary. I will generate a pseudo-document with geographical context to improve vector recall."
  - pseudo_document: "Arizona, Maricopa County, Tempe, postal codes geography, US states, Southwest region."

User's query: "hey can u tell me how to fix physics crash returnal"
Expected Action: Conversational noise, poor syntax, lacks technical terms. Needs rewrite AND expansion.
Tools to call (Parallel):
- rewrite_query
  - reasoning: "Removing conversational noise and formalizing syntax for search."
  - rewritten_query: "How to resolve physics engine crashes in the game Returnal"
- expand_query
  - reasoning: "Adding technical gaming context to expand the search surface."
  - pseudo_document: "Unreal Engine 4, PS5, PC port, Housemarque, collision bug, patch update, fatal error, graphics drivers, GPU crash."
"""

    try:
        # L'LLM processerà il ragionamento e chiamerà i tool appropriati
        response_msg = llm_with_tools.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User's query: '{current_query}'. Analyze the information need and call the appropriate tools.")
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
        for tool_call in response_msg.tool_calls:
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
            final_queries = [f"{q} \n[Contesto Espanso]: {expansion_text}" for q in base_queries]
        else:
            final_queries = base_queries
            
        # Uniamo le query in una singola stringa perché GraphState.current_query è di tipo str
        final_query_str = "\n---\n".join(final_queries)
            
        print(f"[REFINER] Tool applicati: {', '.join(tools_used).upper()}")
        print(f"[REFINER] Output finale:\n{final_query_str}")
        
        return {
            "current_query": final_query_str
        }
    else:
        print("[REFINER] L'agente ha ritenuto la query perfetta. Nessun tool chiamato.")
        return {"current_query": current_query}