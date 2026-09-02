import json
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from graph import GraphState
import os
import requests
from bs4 import BeautifulSoup

# Google Search API: Richiede GOOGLE_API_KEY e GOOGLE_CSE_ID settati come variabili d'ambiente.
google_wrapper = GoogleSearchAPIWrapper(k=5)

@tool
def web_search(query: str) -> str:
    """
    Executes a web search on Google. Use this to find snippets and links.
    Argument: Natural language search query.
    """
    try:
        results = google_wrapper.results(query, num_results=5)
        clean_text = ""
        for i, res in enumerate(results):
            clean_text += f"\n{i+1}. Titolo: {res.get('title')}\n   Snippet: {res.get('snippet')}\n   Link: {res.get('link')}\n"
        return clean_text if clean_text else "Nessun risultato sul web."
    except Exception as e:
        return f"Errore: {str(e)}"

@tool
def scrape_website(url: str) -> str:
    """
    Downloads and extracts the text content of a webpage. Use this ONLY when the snippet from web_search is not enough.
    Argument: The exact URL (link) to scrape.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract text from paragraphs
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        # Limit to 3000 chars to avoid blowing up context window
        return text[:3000] if text else "Nessun testo leggibile trovato."
    except Exception as e:
        return f"Errore nello scraping: {str(e)}"

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
                                            t_name = "web_search"
                                        elif "url" in t_args:
                                            t_name = "scrape_website"
                                    
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
    tools = [web_search, scrape_website]
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

STRATEGY OPTIONS:
- 'internal_knowledge': For trivial facts (math, capitals) where NO search is needed.
- 'single_retrieval': For straightforward questions answerable with one search. DO NOT use this for comparisons.
- 'multi_step': For complex queries needing iterative reasoning, multiple sources, OR comparing two distinct entities (e.g., "Are X and Y the same...").

CRITICAL RULES:
- You do not support native function calling. You MUST manually output a RAW JSON object calling the 'AdaptiveStrategy' tool.
- DO NOT output any conversational text before or after the JSON.
- MUST use ONLY ENGLISH. No Chinese characters.

--- FEW-SHOT EXAMPLES ---

User query: "What is 2+2?"
```json
{{
  "name": "AdaptiveStrategy",
  "arguments": {{
    "reasoning": "This is a basic math question that does not require any external search.",
    "strategy": "internal_knowledge",
    "search_term": ""
  }}
}}
```
User query: "Who won the World Cup in 2022?"
```json
{{
  "name": "AdaptiveStrategy",
  "arguments": {{
    "reasoning": "This is a factual question that requires a single lookup to verify.",
    "strategy": "single_retrieval",
    "search_term": "winner of World Cup 2022"
  }}
}}
```

User query: "When was [CEO of Apple] born?" (Assuming context says CEO is Tim Cook)
```json
{{
  "name": "AdaptiveStrategy",
  "arguments": {{
    "reasoning": "I need to search for Tim Cook's birth date. A single search is enough.",
    "strategy": "single_retrieval",
    "search_term": "Tim Cook birth date"
  }}
}}
```
CRITICAL: DO NOT copy these examples verbatim. Formulate your search term based on the ACTUAL current query and context."""
        llm_strategy = llm.bind_tools([AdaptiveStrategy])
        resp_strat = llm_strategy.invoke([HumanMessage(content=f"{strategy_prompt}\n\nUser query: '{query}'")])
        
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
            # Pulizia dei tag <think> per evitare crash della KV cache nei nodi successivi
            import re
            cleaned_ans = re.sub(r'<think>.*?</think>', '', getattr(ans, "content", str(ans)), flags=re.DOTALL).strip()
            step_context += f"Internal Knowledge: {cleaned_ans}\n"
            
        elif strategy == "single_retrieval":
            print(f"   🔎 [PLANNER] Eseguo Single Retrieval per -> '{search_term}'")
            tool_result = web_search.invoke({"query": search_term})
            preview = str(tool_result)[:150].replace('\n', ' ') + "..."
            print(f"      📄 [PLANNER] Retrieved: {preview}")
            step_context += f"Search Result ({search_term}): {tool_result}\n"
            
        else:
            # MULTI-STEP REACT
            system_prompt = f"""You are a Multi-Step Research Agent.
Evaluate the remaining query using the available tools:
- 'web_search': To search Google and get snippets with links.
- 'scrape_website': To extract the full text of a specific URL (only if the snippet was insufficient).

Rules:
1. If you need information, CALL 'web_search' first. If a snippet looks promising but lacks detail, use 'scrape_website' on the link.
2. If the retrieved context already answers the query, format your response as EXACTLY:
FINAL ANSWER: [your complete answer here]
3. DO NOT hallucinate. You MUST base your answer on the retrieved text.

Example of Calling a Tool:
Thought: I need to search for Shirley Temple to find out when she was born.
```json
{{
  "name": "web_search",
  "arguments": {{"query": "Shirley Temple birth date"}}
}}
```

PREVIOUSLY ACQUIRED CONTEXT:
{global_context if global_context else "No previous context."}

INSTRUCTIONS:
1. If the context lacks the answer, first write a 'Thought: ' explaining your reasoning, then output the JSON tool call to search.
2. If the context has the answer, DO NOT call tools. Answer with FINAL ANSWER: [answer].
3. CRITICAL: Output ONLY English. No Chinese characters.
4. FORMAT: You MUST manually output a RAW JSON object representing the tool call immediately after your Thought.
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
                            if tool_call['name'] == "web_search":
                                print(f"   🌐 [PLANNER] Azione ReAct: Eseguo web_search per -> '{search_query}'")
                                tool_result = web_search.invoke({"query": search_query})
                            elif tool_call['name'] == "scrape_website":
                                target_url = tool_call['args'].get('url', search_query) # fallback se usa 'query' invece di 'url'
                                print(f"   🌐 [PLANNER] Azione ReAct: Eseguo scraping su -> '{target_url}'")
                                tool_result = scrape_website.invoke({"url": target_url})
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