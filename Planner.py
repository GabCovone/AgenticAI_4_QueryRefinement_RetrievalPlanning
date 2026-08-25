import json
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from graph import GraphState

# --- 1. SETUP DEL RETRIEVER (DuckDuckGo API) ---
# Forziamo i risultati in inglese per evitare snippet di Wikipedia in altre lingue
wrapper = DuckDuckGoSearchAPIWrapper(region="us-en")
ddg_search = DuckDuckGoSearchResults(api_wrapper=wrapper, max_results=3)

@tool
def web_search(query: str) -> str:
    """
    Executes a web search to retrieve real-time and up-to-date information.
    Argument: The optimized search string to send to the search engine.
    Returns: One or more text snippets retrieved from English web pages.
    """
    try:
        results = ddg_search.invoke({"query": query})
        return results
    except Exception as e:
        return f"Errore durante la ricerca: {str(e)}"

class AdaptiveStrategy(BaseModel):
    """USE THIS TOOL to classify the query strategy."""
    reasoning: str = Field(description="Reasoning for selecting the strategy.")
    strategy: Literal["internal_knowledge", "single_retrieval", "multi_step"] = Field(description="The chosen strategy.")
    search_term: str = Field(description="If single_retrieval, the exact optimized query to search. Resolve any placeholders (e.g. [name]) using the context.")

def extract_tool_calls(response_msg):
    tool_calls_list = []
    if hasattr(response_msg, "tool_calls") and response_msg.tool_calls:
        tool_calls_list = response_msg.tool_calls
    else:
        import json
        content = getattr(response_msg, "content", str(response_msg))
        
        # NUOVO PARSER SUPER-ROBUSTO a finestra scorrevole
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
                                args = parsed.get("arguments", parsed.get("args", parsed))
                                tool_calls_list.append({"name": parsed["name"], "args": args, "id": "fb_123"})
                        except Exception:
                            pass
                        i = j # Salta i caratteri elaborati e vai oltre
                        break
                        
                    if char == '\\':
                        escape = not escape
                    else:
                        escape = False
            i += 1
    return tool_calls_list


# --- 2. NODO PLANNER ---
def planner_node(state: GraphState, llm) -> dict:
    """
    Esegue il Multi-Step Retrieval Planning (ReAct + IRCoT).
    Gestisce in sequenza le query generate dal Refiner e sintetizza la risposta finale.
    """
    print("\n--- [PLANNER] Inizio Esecuzione Piano Multi-Step ---")
    
    queries: List[str] = state.get("current_query", [])
    if isinstance(queries, str):
        queries = [q.strip() for q in queries.split("\n---\n") if q.strip()]
        
    global_context = ""
    tools = [web_search]
    llm_with_tools = llm.bind_tools(tools)
    
    # Limite di sicurezza per evitare OOM (Out Of Memory) su Colab
    MAX_STEPS_PER_QUERY = 2 
    
    # --- FASE A: RETRIEVAL INTERATTIVO (IRCoT, Adaptive-RAG & ReAct Loop) ---
    for i, query in enumerate(queries):
        print(f"\n[PLANNER] Step {i+1}/{len(queries)}: Risoluzione della query -> '{query}'")
        step_context = ""
        
        # 1. ADAPTIVE-RAG: Classificazione della strategia
        print("[PLANNER] [Adaptive-RAG] Valutazione della strategia...")
        strategy_prompt = f"""You are a Strategic Planner. Your task is to classify the query to decide the best retrieval strategy.

PREVIOUSLY ACQUIRED CONTEXT (Use this to resolve placeholders like [name] in the query):
{global_context if global_context else "No previous context."}

Options:
- 'internal_knowledge': Use for trivial factual questions where no search is needed (e.g. math, capitals).
- 'single_retrieval': Use when the query needs external information but is straightforward and can be answered with one search.
- 'multi_step': Use when the query is complex, requires iterative reasoning, or needs information from multiple sources.

You can think out loud first (keep it extremely brief), but you MUST conclude your response with a VALID JSON BLOCK calling the 'AdaptiveStrategy' tool.

--- FEW-SHOT EXAMPLES ---

User query: "What is 2+2?"
{{
  "name": "AdaptiveStrategy",
  "arguments": {{
    "reasoning": "This is a basic math question that does not require any external search.",
    "strategy": "internal_knowledge",
    "search_term": ""
  }}
}}

User query: "Who won the World Cup in 2022?"
{{
  "name": "AdaptiveStrategy",
  "arguments": {{
    "reasoning": "This is a factual question that requires a single lookup to verify.",
    "strategy": "single_retrieval",
    "search_term": "winner of World Cup 2022"
  }}
}}

User query: "In which city was [director's name] born?" (Assuming context says director is Christopher Nolan)
{{
  "name": "AdaptiveStrategy",
  "arguments": {{
    "reasoning": "I need to search for Christopher Nolan's birthplace. A single search is enough.",
    "strategy": "single_retrieval",
    "search_term": "Christopher Nolan birthplace"
  }}
}}"""
        llm_strategy = llm.bind_tools([AdaptiveStrategy])
        resp_strat = llm_strategy.invoke([SystemMessage(content=strategy_prompt), HumanMessage(content=f"User query: '{query}'")])
        
        t_calls = extract_tool_calls(resp_strat)
        strategy = "multi_step" # Default in caso di errore
        search_term = query
        if t_calls and t_calls[0]["name"] == "AdaptiveStrategy":
            strategy = t_calls[0]["args"].get("strategy", "multi_step")
            search_term = t_calls[0]["args"].get("search_term", query)
            if not search_term: search_term = query
            print(f"[PLANNER] Strategia selezionata: {strategy.upper()}")
        else:
            print("[PLANNER] Strategia di fallback: MULTI_STEP")

        # 2. ESECUZIONE DELLA STRATEGIA
        if strategy == "internal_knowledge":
            prompt = f"PREVIOUS CONTEXT:\n{global_context if global_context else 'None'}\n\nQuestion: {query}\n\nAnswer briefly (use the context to resolve placeholders like [person]):"
            ans = llm.invoke([HumanMessage(content=prompt)])
            step_context += f"Internal Knowledge: {ans.content}\n"
            
        elif strategy == "single_retrieval":
            print(f"[PLANNER] Eseguo Single Retrieval per -> '{search_term}'")
            tool_result = web_search.invoke({"query": search_term})
            step_context += f"Search Result ({search_term}): {tool_result}\n"
            
        else:
            # MULTI-STEP REACT
            system_prompt = f"""You are a Multi-Step Research Agent.
Your task is to find information to answer the current query. 
You have access to the 'web_search' tool. Use it to search the internet.

PREVIOUSLY ACQUIRED CONTEXT (Use this to guide your reasoning):
{global_context if global_context else "No previous context."}

INSTRUCTIONS:
1. Analyze the query. If you do not have the information in the context, CALL THE 'web_search' TOOL.
2. If you already have enough information in the context, DO NOT call tools and answer with a concluding thought.

You can think out loud first (keep it extremely brief), but you MUST conclude your response with VALID JSON BLOCKS calling the tools.

--- FEW-SHOT EXAMPLES ---

Query: "Birthplace of Christopher Nolan"
Context: "No previous context."
{{
  "name": "web_search",
  "arguments": {{"query": "Christopher Nolan birthplace"}}
}}
"""
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Current query: {query}")
            ]
            
            for attempt in range(MAX_STEPS_PER_QUERY):
                try:
                    response_msg = llm_with_tools.invoke(messages)
                    messages.append(response_msg)
                    
                    t_calls_react = extract_tool_calls(response_msg)
                    if t_calls_react:
                        for tool_call in t_calls_react:
                            if tool_call['name'] == "web_search":
                                search_query = tool_call['args'].get('query', query)
                                print(f"[PLANNER] Azione ReAct: Eseguo ricerca web per -> '{search_query}'")
                                tool_result = web_search.invoke({"query": search_query})
                                messages.append(ToolMessage(
                                    content=str(tool_result),
                                    name=tool_call['name'],
                                    tool_call_id=tool_call.get('id', 'fb_123')
                                ))
                                step_context += f"Search Result ({search_query}): {tool_result}\n"
                    else:
                        print("[PLANNER] Ragionamento concluso per questo step.")
                        break
                        
                except Exception as e:
                    print(f"[PLANNER ERROR] Errore nel loop di ricerca: {e}")
                    break
                    
        global_context += step_context
        print(f"[PLANNER] Contesto aggiornato con i risultati dello step {i+1}.")

    # --- FASE B: SINTESI FINALE ---
    print("\n[PLANNER] Generazione Sintesi Finale basata sui documenti recuperati...")
    
    synthesis_prompt = """You are the Final Synthesis Agent.
Your task is to answer the original user question using ONLY the provided retrieved context.
If the context does not contain sufficient information, admit that you do not have the answer. Do not hallucinate or invent data.
Provide a direct, extremely concise, and brief response without unnecessary conversational filler. Keep your internal reasoning short.
You MUST output a textual response. Do not leave the response blank."""

    original_query = state.get("original_query", str(queries))
    
    try:
        messages = [
            SystemMessage(content=synthesis_prompt),
            HumanMessage(content=f"Original Query: '{original_query}'\nRetrieved Context:\n{global_context}")
        ]
        synthesis_response = llm.invoke(messages)
        final_answer = getattr(synthesis_response, "content", str(synthesis_response)).strip()
        
        if not final_answer:
            print("[PLANNER ERROR] L'LLM ha restituito una risposta vuota. Inserimento fallback.")
            final_answer = "Fallback: Unable to generate synthesis due to an unexpected LLM blank response."
    except Exception as e:
        print(f"[PLANNER ERROR] Errore in generazione sintesi: {e}")
        final_answer = "Errore durante la generazione della risposta finale."

    print("[PLANNER] Task completato. Passo i dati al Validatore.")
    
    # Ritorna lo stato aggiornato
    return {
        "retrieved_context": global_context,
        "final_answer": final_answer,
        "num_planning": state.get("num_planning", 0) + 1
    }