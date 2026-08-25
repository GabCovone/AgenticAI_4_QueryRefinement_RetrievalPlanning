import os
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from graph import GraphState

# --- COSTANTI DI LIMITE ---
MAX_REFINEMENT = 5
MAX_PLANNING = 5

# --- SCHEMA DI DECISIONE FINALE (CON SELF-RAG, ADAPTIVE-RAG E REFLEXION) ---
class ValidatorDecision(BaseModel):
    """USE THIS TOOL to issue your final routing decision."""
    reflection: str = Field(description="REFLEXION: Critique past iterations (Refinement/Planning Count). Why did previous steps fail? How can we avoid loops?")
    query_complexity: Literal["simple", "complex", "already_decomposed"] = Field(description="ADAPTIVE-RAG: Assess the current query's complexity.")
    is_context_relevant: bool = Field(description="SELF-RAG: Is the retrieved context semantically relevant to the query?")
    is_query_answered: bool = Field(description="SELF-RAG: Does the context fully and completely answer the Original Query?")
    reasoning: str = Field(description="CHAIN OF THOUGHT: Combine the critiques above to justify your routing decision.")
    feedback: str = Field(description="Instructions for the next agent (e.g. what keywords to use, what to search next). Empty if finish.")
    next_action: Literal["route_to_refinement", "route_to_planning", "finish"] = Field(
        description="The next node to send the execution to."
    )

# --- LOGICA DEL NODO VALIDATORE ---
def validator_node(state: GraphState, llm) -> dict:
    query = state.get("current_query", state.get("original_query", ""))
    context = state.get("retrieved_context", "")
    num_ref = state.get("num_refinement", 0)
    num_plan = state.get("num_planning", 0)
    
    if num_ref >= MAX_REFINEMENT and num_plan >= MAX_PLANNING:
        print("[VALIDATOR] Limiti massimi raggiunti.")
        return {"next_node": "finish", "feedback_history": ["Forced finish due to limits."]}

    original_query = state.get("original_query", str(query))

    system_prompt = f"""You are the Orchestrator Validator Agent of an Advanced Information Retrieval system.
Your job is to apply Reflexion, Adaptive-RAG, and Self-RAG to route execution between the Query Refiner and the Planner.

1. REFLEXION: Look at the Refinement Count and Planning Count. If they are > 0, it means past attempts failed. Reflect on WHY they failed before deciding.
2. ADAPTIVE-RAG (Routing based on complexity):
   - If Context is empty and query is 'complex' (multi-hop, multiple subjects) -> route to 'route_to_refinement'.
   - If Context is empty and query is 'simple' or 'already_decomposed' (contains '---') -> route to 'route_to_planning'.
3. SELF-RAG (Critique of retrieval):
   - Evaluate `is_context_relevant` (Did the Planner find good documents?).
   - Evaluate `is_query_answered` (Is the exact answer to the Original Query present?).
   - If `is_query_answered` is true -> route to 'finish'.
   - If context is irrelevant or incomplete, give actionable `feedback` and route to 'route_to_refinement' (to rewrite query) or 'route_to_planning' (to search deeper).

You MUST output EXACTLY AND ONLY a valid JSON object calling the ValidatorDecision tool.
CRITICAL RULE: DO NOT TRANSLATE THE JSON KEYS! You must strictly use the exact English keys: "reflection", "query_complexity", "is_context_relevant", "is_query_answered", "reasoning", "feedback", "next_action". Do not output Chinese characters like "方面".

--- FEW-SHOT EXAMPLE ---
{{
  "name": "ValidatorDecision",
  "arguments": {{
    "reflection": "The planner ran once but found irrelevant info. We need to change the search terms.",
    "query_complexity": "already_decomposed",
    "is_context_relevant": false,
    "is_query_answered": false,
    "reasoning": "The retrieved documents talk about the wrong person. The Refiner should add more specific keywords.",
    "feedback": "Add the keyword 'director' to the query to disambiguate.",
    "next_action": "route_to_refinement"
  }}
}}
"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Original Query: '{original_query}'.\nCurrent Query: '{query}'.\nCurrent Context: '{context}'\nRefinement Count: {num_ref}\nPlanning Count: {num_plan}\n\nAnalyze the state and output the JSON tool call.")
    ]
    
    llm_with_tools = llm.bind_tools([ValidatorDecision])
    decision_obj = None

    print(f"\n[VALIDATOR] Avvio ciclo decisionale (Ref={num_ref}, Plan={num_plan})...")
    
    response = llm_with_tools.invoke(messages)
    
    # Tentativo di estrazione del JSON dalla risposta
    import json
    
    content = getattr(response, "content", str(response))
    
    # Estrai tutti gli oggetti JSON validi
    brace_level = 0
    current_json = ""
    in_string = False
    escape = False
    
    for char in content:
        if char == '"' and not escape:
            in_string = not in_string
        if not in_string:
            if char == '{':
                brace_level += 1
            elif char == '}':
                brace_level -= 1
        
        if brace_level > 0 or (char == '}' and brace_level == 0 and current_json):
            current_json += char
            if brace_level == 0:
                try:
                    parsed = json.loads(current_json)
                    if "name" in parsed:
                        args = parsed.get("arguments", parsed.get("args", parsed))
                        decision_obj = ValidatorDecision(**args)
                        print("[VALIDATOR] Fallback parsing riuscito!")
                        break # Abbiamo trovato la decisione
                except Exception:
                    pass
                current_json = ""
        elif char == '\\':
            escape = not escape
        else:
            escape = False
            
    if not decision_obj:
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_call = response.tool_calls[0]
            if tool_call["name"] == "ValidatorDecision":
                decision_obj = ValidatorDecision(**tool_call["args"])
                print("[VALIDATOR] Tool call nativo riuscito!")

    if not decision_obj:
        print(f"[VALIDATOR] ERRORE: Qwen non ha restituito un JSON valido. Contenuto: {content}")
        decision_obj = ValidatorDecision(
            reflection="Fallback triggered due to invalid JSON.",
            query_complexity="complex",
            is_context_relevant=False,
            is_query_answered=False,
            reasoning="Fallback error: no valid JSON.", 
            feedback="", 
            next_action="finish"
        )

    # 4. OVERRIDE DI SICUREZZA
    final_action = decision_obj.next_action
    if final_action == "route_to_refinement" and num_ref >= MAX_REFINEMENT:
        final_action = "route_to_planning"
    elif final_action == "route_to_planning" and num_plan >= MAX_PLANNING:
        final_action = "finish"

    print(f"[VALIDATOR] Reasoning di Qwen: {decision_obj.reasoning}")
    print(f"[VALIDATOR] Instradamento verso: {final_action}")

    return {
        "next_node": final_action,
        "feedback_history": [decision_obj.feedback]
    }