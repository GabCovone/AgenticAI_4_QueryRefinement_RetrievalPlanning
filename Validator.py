import os
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from graph import GraphState

# --- COSTANTI DI LIMITE ---
MAX_REFINEMENT = 5
MAX_PLANNING = 5

# --- TOOL DELLE METRICHE ---
@tool
def calc_refinement_metrics(query: str) -> dict:
    """Calculates refinement metrics. Returns 'ambiguity_score' and 'breadth_index'."""
    print(f"\n   [TOOL ESEGUITO DA QWEN] -> calc_refinement_metrics per: '{query}'")
    return {"ambiguity_score": 4, "breadth_index": 7}

@tool
def calc_planning_metrics(query: str, retrieved_context: str) -> dict:
    """Calculates planning metrics. Returns 'information_sufficiency' and 'consistency_score'."""
    print(f"\n   [TOOL ESEGUITO DA QWEN] -> calc_planning_metrics")
    return {"information_sufficiency": 8, "consistency_score": 9}

# --- SCHEMA DI DECISIONE FINALE (Funge da Tool per terminare il loop) ---
class ValidatorDecision(BaseModel):
    """USE THIS TOOL ONLY AT THE END to issue your final routing decision."""
    reasoning: str = Field(description="Your reasoning based on the results of the previously called metric tools.")
    feedback: str = Field(description="Instructions or feedback for the next agent.")
    next_action: Literal["route_to_refinement", "route_to_planning", "finish"] = Field(
        description="The next node to send the execution to."
    )

# --- LOGICA DEL NODO VALIDATORE ---
def validator_node(state: GraphState, llm) -> dict:
    query = state.get("current_query", state["original_query"])
    context = state.get("retrieved_context", "")
    num_ref = state.get("num_refinement", 0)
    num_plan = state.get("num_planning", 0)
    
    # 1. HARD LIMIT CHECK
    if num_ref >= MAX_REFINEMENT and num_plan >= MAX_PLANNING:
        print("[VALIDATOR] Limiti massimi raggiunti.")
        return {"next_node": "finish", "feedback_history": ["Forced finish due to limits."]}

    # 2. PREPARAZIONE DEL MODELLO E DEI MESSAGGI
    system_prompt = f"""You are the Validator Agent of an Information Retrieval system.
You must decide if the query requires further refinement or if it can be routed to the planner.
WARNING: Do not make decisions blindly! You must ALWAYS use tools.

--- FEW-SHOT EXAMPLES ---

User's Input: Current Query: 'What is X?'. Current Context: ''
Expected Action: No context retrieved yet. Check refinement metrics.
Tools to call:
- calc_refinement_metrics
  - query: "What is X?"

User's Input: Current Query: 'What is X?'. Current Context: 'Context about X...'
Expected Action: Context retrieved. Check planning metrics.
Tools to call:
- calc_planning_metrics
  - query: "What is X?"
  - retrieved_context: "Context about X..."

After evaluating metrics, make a final decision.
Tools to call:
- ValidatorDecision
  - reasoning: "The metrics indicate high ambiguity. We need more refinement."
  - feedback: "Please decompose the query."
  - next_action: "route_to_refinement"
"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Current Query: '{query}'.\nCurrent Context: '{context}'\n\nAnalyze the state and call the appropriate tool.")
    ]
    
    # Bindiamo TUTTI i tool al modello, compreso lo schema finale Pydantic
    llm_with_tools = llm.bind_tools([calc_refinement_metrics, calc_planning_metrics, ValidatorDecision])
    
    decision_obj = None

    # 3. IL VERO LOOP AGENTICO (Tool Calling)
    print(f"\n[VALIDATOR] Avvio ciclo decisionale (Ref={num_ref}, Plan={num_plan})...")
    
    while True:
        # L'LLM decide autonomamente cosa fare in base allo storico dei messaggi
        response = llm_with_tools.invoke(messages)
        messages.append(response) # Aggiungiamo la risposta (che contiene la richiesta del tool) alla memoria
        
        # Se non chiama tool, c'è un errore logico (forziamo l'uscita o gestiamo l'errore)
        if not response.tool_calls:
            print("[VALIDATOR] ERRORE: Qwen non ha usato alcun tool.")
            decision_obj = ValidatorDecision(
                reasoning="Fallback error: no tools called.", 
                feedback="", 
                next_action="finish"
            )
            break
            
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # Se l'LLM ha deciso di chiamare il tool finale, rompiamo il ciclo
        if tool_name == "ValidatorDecision":
            print(f"[VALIDATOR] Qwen ha preso la decisione finale!")
            decision_obj = ValidatorDecision(**tool_args)
            break
            
        # Altrimenti, l'LLM ha richiesto le metriche. Noi le calcoliamo e gliele restituiamo.
        elif tool_name == "calc_refinement_metrics":
            result = calc_refinement_metrics.invoke(tool_args)
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
            
        elif tool_name == "calc_planning_metrics":
            result = calc_planning_metrics.invoke(tool_args)
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
            
        else:
            # Rete di sicurezza nel caso si inventi un tool inesistente
            messages.append(ToolMessage(content="Errore: Tool inesistente.", tool_call_id=tool_call["id"]))

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