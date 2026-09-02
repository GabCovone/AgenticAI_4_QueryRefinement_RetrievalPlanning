import json
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
# from langchain_community.tools import DuckDuckGoSearchResults
# from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from graph import GraphState
import os
import requests
# from bs4 import BeautifulSoup

from local_kb import search_index

@tool
def local_kb_search(query: str) -> str:
    """
    Executes a search on the local Wikipedia inverted index (BM25).
    Argument: Natural language search query.
    """
    try:
        return search_index(query, top_k=3)
    except Exception as e:
        return f"Errore locale: {str(e)}"

# @tool
# def scrape_website(url: str) -> str:
#     """
#     Downloads and extracts the text content of a webpage. Use this ONLY when the snippet from web_search is not enough.
#     Argument: The exact URL (link) to scrape.
#     """
#     try:
#         headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
#         response = requests.get(url, headers=headers, timeout=5)
#         response.raise_for_status()
#         soup = BeautifulSoup(response.text, 'html.parser')
#         # Extract text from paragraphs
#         text = ' '.join([p.get_text() for p in soup.find_all('p')])
#         # Limit to 3000 chars to avoid blowing up context window
#         return text[:3000] if text else "Nessun testo leggibile trovato."
#     except Exception as e:
#         return f"Errore nello scraping: {str(e)}"

class AdaptiveStrategy(BaseModel):
    """USE THIS TOOL to classify the query strategy."""
    reasoning: str = Field(description="Reasoning for selecting the strategy.")
    strategy: Literal["internal_knowledge", "single_retrieval", "multi_step"] = Field(description="The chosen strategy.")
    search_term: str = Field(description="If single_retrieval, the exact optimized query to search. Resolve any placeholders (e.g. [name]) using the context.")

def extract_tool_calls(response_msg):
    tool_calls_list = []
    # Usa tool_calls nativi solo se hanno argomenti non vuoti
    if hasattr(response_msg, "tool_calls") and response_msg.tool_calls and all(tc.get("args") for tc in response_msg.tool_calls):
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
                                json_str = content[i:j+1]
                                try:
                                    parsed = json.loads(json_str)
                                    t_name = None
                                    t_args = {}
                                    if "name" in parsed:
                                        t_name = parsed["name"]
                                        t_args = parsed.get("arguments", parsed.get("args", parsed.get("parameters", parsed)))
                                    else:
                                        # Infer name from keys if 'name' wrapper is missing
                                        t_args = parsed
                                        if "strategy" in t_args or "search_term" in t_args:
                                            t_name = "AdaptiveStrategy"
                                        elif "query" in t_args:
                                            t_name = "local_kb_search"
                                    
                                    if t_name:
                                        tool_calls_list.append({"name": t_name, "args": t_args, "id": "fb_123"})
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
    tools = [local_kb_search]
    llm_with_tools = llm.bind_tools(tools)
    
    # Limite di sicurezza per evitare OOM (Out Of Memory) su Colab
    MAX_STEPS_PER_QUERY = 3
    
    # --- FASE A: RETRIEVAL INTERATTIVO (IRCoT, Adaptive-RAG & ReAct Loop) ---
    for i, query in enumerate(queries):
        print(f"\n[PLANNER] Step {i+1}/{len(queries)}: Risoluzione della query -> '{query}'")
        step_context = ""
        
        # 1. ADAPTIVE-RAG: Classificazione della strategia
        print("[PLANNER] [Adaptive-RAG] Valutazione della strategia...")
        strategy_prompt = f"""You are a Strategic Planner. Classify the query to decide the best retrieval strategy.

PREVIOUSLY ACQUIRED CONTEXT (Use this to resolve placeholders like [name]):
{global_context if global_context else "No previous context."}

Query: '{query}'

You have access to the tool 'AdaptiveStrategy'. Call this tool to select one of the following strategies:
1. 'internal_knowledge': Only if the answer is trivial and globally known (e.g., "capital of France").
2. 'single_retrieval': Only if the query is a simple, atomic entity search (e.g., "when was X born?"). Provide the exact optimized search_term for the local kb.
3. 'multi_step': For any complex, multi-hop, or sequential query.

Wait for the system to process the strategy. Do not output anything else.
"""
        try:
            # Invoco l'LLM forzandolo (se supportato) o suggerendogli l'uso del tool AdaptiveStrategy
            strat_msg = llm_with_tools.invoke([SystemMessage(content=strategy_prompt)])
            
            # Estrazione sicura del tool call
            t_calls = extract_tool_calls(strat_msg)
            
            strategy = "multi_step"
            search_term = query
            
            if t_calls and t_calls[0]["name"] == "AdaptiveStrategy":
                args = t_calls[0]["args"]
                strategy = args.get("strategy", "multi_step")
                search_term = args.get("search_term", query)
            
            print(f"[PLANNER] Strategia selezionata: {strategy.upper()}")
            
            # Esecuzione condizionale basata sulla strategia
            if strategy == "internal_knowledge":
                print(f"[PLANNER] Uso conoscenza interna per: '{query}'")
                step_context = f"Answer internally known. Query: {query}\n"
                
            elif strategy == "single_retrieval":
                print(f"[PLANNER] Eseguo single_retrieval per -> '{search_term}'")
                # Chiamata DIRETTA al tool
                tool_result = local_kb_search.invoke({"query": search_term})
                
                # Preview sicura del risultato
                preview = str(tool_result)[:150].replace('\n', ' ') + "..."
                print(f"      📄 [PLANNER] Retrieved: {preview}")
                
                step_context = f"Search Result ({search_term}): {tool_result}\n"
                
            else:
                # 2. MULTI-STEP RETRIEVAL (ReAct Loop)
                print("[PLANNER] Avvio ciclo ReAct (Multi-Step)...")
                system_prompt = f"""You are an advanced investigative AI agent.
Your task is to answer the query by using the local_kb_search tool to gather evidence from the Wikipedia Knowledge Base.

AVAILABLE TOOLS:
1. local_kb_search(query: str): Searches the local Wikipedia BM25 index and returns relevant passages. Use this to find facts.

PREVIOUSLY ACQUIRED CONTEXT:
{global_context if global_context else "No previous context."}

INSTRUCTIONS:
1. If the context lacks the answer, first write a 'Thought: ' explaining your reasoning, then output EXACTLY ONE JSON tool call to search. DO NOT output multiple JSON blocks.
2. If the context has the answer, DO NOT call tools. Answer with FINAL ANSWER: [answer].
3. CRITICAL: Output ONLY English. No Chinese characters.
4. FORMAT: You MUST manually output a RAW JSON object representing the tool call immediately after your Thought.
Example:
Thought: I need to find the capital of France.
{{
  "name": "local_kb_search",
  "args": {{"query": "capital of France"}}
}}
"""
                messages = [
                    HumanMessage(content=f"{system_prompt}\n\nQuery to answer: '{query}'")
                ]
                
                for attempt in range(MAX_STEPS_PER_QUERY):
                    try:
                        response_msg = llm_with_tools.invoke(messages)
                        messages.append(response_msg)
                        
                        t_calls_react = extract_tool_calls(response_msg)
                        
                        content_str = getattr(response_msg, "content", "").strip()
                        
                        if t_calls_react:
                            observations_text = ""
                            for tool_call in t_calls_react:
                                search_query = tool_call['args'].get('query', query)
                                if tool_call['name'] == "local_kb_search":
                                    print(f"   🌐 [PLANNER] Azione ReAct: Eseguo local_kb_search per -> '{search_query}'")
                                    tool_result = local_kb_search.invoke({"query": search_query})
                                else:
                                    tool_result = "Tool sconosciuto."
                                # Print a preview of the retrieved text
                                preview = str(tool_result)[:150].replace('\n', ' ') + "..."
                                print(f"      📄 [PLANNER] Retrieved: {preview}")
                                
                                observations_text += f"Observation from tool '{tool_call['name']}' (query: '{search_query}'): {tool_result}\n\n"
                                step_context += f"Search Result ({search_query}): {tool_result}\n"
                        
                            # Append a SINGLE message for all observations to maintain user/assistant alternation
                            if hasattr(response_msg, "tool_calls") and response_msg.tool_calls:
                                # If it was native tool calls, we need a ToolMessage for EACH call
                                for idx, tool_call in enumerate(t_calls_react):
                                    messages.append(ToolMessage(
                                        content=observations_text if idx==0 else "See previous ToolMessage",
                                        name=tool_call['name'],
                                        tool_call_id=tool_call.get('id', f'fb_{idx}')
                                    ))
                            else:
                                observations_text += "Analyze these observations. If you have the answer, output FINAL ANSWER: [answer]. Otherwise, write a Thought and output another JSON tool call."
                                messages.append(HumanMessage(content=observations_text))
                        else:
                            if content_str:
                                # Rimuoviamo il tag json residuo se presente
                                clean_thought = content_str.replace("```json", "").replace("```", "").strip()
                                print(f"   💭 [PLANNER] Reasoning Conclusivo: {clean_thought[:200]}...")
                            else:
                                print("   💭 [PLANNER] Ragionamento concluso per questo step.")
                            break
                            
                    except Exception as e:
                        print(f"[PLANNER ERROR] Errore nel loop di ricerca: {e}")
                        break
                        
        except Exception as e:
            print(f"[PLANNER ERROR] Errore in strategy_prompt: {e}")

        global_context += step_context
        print(f"[PLANNER] Contesto aggiornato con i risultati dello step {i+1}.")

    # --- FASE B: SINTESI FINALE ---
    print("\n[PLANNER] Generazione Sintesi Finale basata sui documenti recuperati...")
    
    synthesis_prompt = """You are the Final Synthesis Agent.
Your task is to answer the original user question using ONLY the provided retrieved context.
Provide a direct, extremely concise, and brief response without unnecessary conversational filler. Keep your internal reasoning short.

CRITICAL RULES:
1. STRICT GROUNDING: You must explicitly find the exact entities asked in the question within the text. If the retrieved context talks about a different entity, YOU DO NOT HAVE THE ANSWER.
2. NO GUESSING: If the context does not contain sufficient information, reply EXACTLY with: "I do not have enough information." Do not invent data.
3. If the Original Query is a Yes/No question, your final answer MUST start with "yes" or "no" (lowercase) followed by a brief explanation.
4. YOU MUST ONLY USE ENGLISH. Do not use or output any Chinese characters under any circumstances."""

    original_query = state.get("original_query", str(queries))
    
    try:
        synthesis_response = llm.invoke([
            HumanMessage(content=f"{synthesis_prompt}\n\nOriginal Query: '{original_query}'\nRetrieved Context:\n{global_context}")
        ])
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