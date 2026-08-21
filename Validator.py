import os
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from graph import GraphState

# --- COSTANTI DI LIMITE ---
MAX_REFINEMENT = 5
MAX_PLANNING = 5

# --- SCHEMA DI DECISIONE FINALE ---
class ValidatorDecision(BaseModel):
    """USE THIS TOOL to issue your final routing decision."""
    reasoning: str = Field(description="Your reasoning based on the query and context.")
    feedback: str = Field(description="Instructions or feedback for the next agent.")
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

    system_prompt = f"""You are the Validator Agent of an Information Retrieval system.
You must decide if the query requires further refinement, or if it can be routed to the planner, or if the retrieved context already answers the query completely (finish).

RULES:
1. If 'Current Context' is empty, the query has not been searched yet. You must route to 'route_to_refinement' if the query is complex, ambiguous, or needs multi-hop reasoning. Otherwise, route to 'route_to_planning' to execute the search.
2. If 'Current Context' has information but it doesn't answer the query, provide feedback and 'route_to_refinement'.
3. You MUST output your decision by calling the ValidatorDecision tool. Since you are running in a constrained environment, output EXACTLY AND ONLY a valid JSON object matching this schema:

{{
  "name": "ValidatorDecision",
  "arguments": {{
    "reasoning": "your chain of thought",
    "feedback": "instructions for the next node",
    "next_action": "route_to_refinement" | "route_to_planning" | "finish"
  }}
}}
"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Current Query: '{query}'.\nCurrent Context: '{context}'\n\nAnalyze the state and output the JSON tool call.")
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