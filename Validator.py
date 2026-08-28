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
    feedback: str = Field(description="Specific instructions for the next agent. For Refiner: tell it which tool to use (decompose/rewrite/expand). For Planner: suggest exact search terms. Empty if finish.")
    next_action: Literal["route_to_refinement", "route_to_planning", "finish"] = Field(
        description="The next node to send the execution to."
    )

# --- LOGICA DEL NODO VALIDATORE ---
def validator_node(state: GraphState, llm) -> dict:
    num_ref = state.get("num_refinement", 0)
    num_plan = state.get("num_planning", 0)
    print(f"\n[VALIDATOR] Avvio ciclo decisionale (Ref={num_ref}, Plan={num_plan})...")
    
    llm_with_tools = llm.bind_tools([ValidatorDecision])
    
    system_prompt = """You are the Semantic Routing Validator.
Evaluate if the current sub-query is ready for retrieval, needs refinement, or is already answered.

ROUTING OPTIONS ('next_action'):
1. 'finish': CHOOSE THIS FIRST if the Current Context or Final Answer completely answers the sub-query. DO NOT route to planning if you already have the answer.
2. 'route_to_planning': Choose this if the sub-query is a SINGLE, ATOMIC question AND the answer is NOT yet in the context. Note: Queries with placeholders (e.g., "[Director of Inception]") ARE atomic and ready for planning.
3. 'route_to_refinement': Choose this if the sub-query is ambiguous, conversational, or a MULTI-HOP question requiring decomposition.

FEEDBACK RULES ('feedback' field):
- For 'route_to_refinement': Explicitly suggest 'decompose_query', 'rewrite_query', or 'expand_query' based on the problem.
- For 'route_to_planning': Suggest specific search keywords.
- For 'finish': Leave empty.

CRITICAL INSTRUCTION: You do not support native function calling. You MUST manually output a RAW JSON object that matches the tool schema. DO NOT output any conversational text before or after the JSON.

--- FEW-SHOT EXAMPLES ---

Example 1 (Ready to finish):
```json
{
  "name": "ValidatorDecision",
  "arguments": {
    "reflection": "The context provides the required information.",
    "query_complexity": "simple",
    "is_context_relevant": true,
    "is_query_answered": true,
    "reasoning": "The query is fully answered by the context.",
    "feedback": "The context successfully answers the question.",
    "next_action": "finish"
  }
}
```

Example 2 (Needs refinement due to empty context or vague query):
```json
{
  "name": "ValidatorDecision",
  "arguments": {
    "reflection": "The context is empty and the query is too conversational.",
    "query_complexity": "complex",
    "is_context_relevant": false,
    "is_query_answered": false,
    "reasoning": "The query needs to be rewritten or decomposed to extract the core entities.",
    "feedback": "Rewrite the query to remove conversational noise and focus on the main entity.",
    "next_action": "route_to_refinement"
  }
}
```

Example 3 (Ready for Planning):
```json
{
  "name": "ValidatorDecision",
  "arguments": {
    "reflection": "The query is specific and well-formed, but the context is empty.",
    "query_complexity": "simple",
    "is_context_relevant": false,
    "is_query_answered": false,
    "reasoning": "Since the query is already clear, it is ready for retrieval.",
    "feedback": "Proceed with document retrieval.",
    "next_action": "route_to_planning"
  }
}
```
"""

    query_raw = state.get("current_query", state.get("original_query", ""))
    queries = [q.strip() for q in query_raw.split("\n---\n") if q.strip()]
    if not queries:
        queries = [query_raw]
        
    context = state.get("retrieved_context", "")
    
    # SE ABBIAMO GIA' UNA FINAL ANSWER, VALUTIAMO SOLO LA QUERY ORIGINALE
    if state.get("final_answer"):
        print("[VALIDATOR] Trovata Final Answer! Valuto se risponde alla query originale...")
        queries = [state.get("original_query", "")]
        context = f"Retrieved Context:\n{context}\n\nProposed Final Answer:\n{state['final_answer']}"
        
    feedbacks = []
    actions = []
    
    for q_idx, q in enumerate(queries):
        print(f"[VALIDATOR] Valutazione sotto-query {q_idx+1}/{len(queries)}: '{q}'")
        try:
            full_queries_context = "\n".join([f"- {sq}" for sq in queries])
            response_msg = llm_with_tools.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"All current sub-queries:\n{full_queries_context}\n\nCurrently evaluating Sub-Query: '{q}'.\nCurrent Context: '{context}'\nRefinement Count: {num_ref}\nPlanning Count: {num_plan}\n\nAnalyze the state for the CURRENT sub-query and output the JSON tool call.")
            ])
            content_resp = getattr(response_msg, "content", str(response_msg))
        except Exception as e:
            print(f"[VALIDATOR ERROR] Errore API per la sub-query: {e}")
            actions.append("route_to_planning")
            continue
            
        import json
        decision_obj = None
        
        # 1. Controlliamo se ha usato tool_calls nativo
        if hasattr(response_msg, "tool_calls") and response_msg.tool_calls:
            t_call = response_msg.tool_calls[0]
            args = t_call.get("args", {})
            safe_args = {
                "reflection": args.get("reflection", "Fallback"),
                "query_complexity": args.get("query_complexity", "complex"),
                "is_context_relevant": args.get("is_context_relevant", False) in [True, "true", "True", 1],
                "is_query_answered": args.get("is_query_answered", False) in [True, "true", "True", 1],
                "reasoning": args.get("reasoning", "Fallback"),
                "feedback": args.get("feedback", ""),
                "next_action": args.get("next_action", "route_to_planning")
            }
            print(f"[VALIDATOR DEBUG] Llama Decision: {safe_args}")
            try:
                decision_obj = ValidatorDecision(**safe_args)
            except Exception as e:
                print(f"[VALIDATOR DEBUG] Pydantic validation failed on tool_calls: {e}")
        
        # 2. Fallback su string parsing
        if not decision_obj:
            i = 0
            while i < len(content_resp):
                if content_resp[i] == '{':
                    brace_level = 0
                    in_string = False
                    escape = False
                    for j in range(i, len(content_resp)):
                        char = content_resp[j]
                        if char == '"' and not escape: in_string = not in_string
                        if not in_string:
                            if char == '{': brace_level += 1
                            elif char == '}': brace_level -= 1
                        if brace_level == 0:
                            try:
                                json_str = content_resp[i:j+1]
                                json_str = json_str.replace(": False", ": false").replace(": True", ": true").replace(":False", ":false").replace(":True", ":true")
                                parsed = json.loads(json_str)
                                if isinstance(parsed, dict):
                                    args = parsed.get("arguments", parsed.get("args", parsed.get("parameters", parsed))) if "name" in parsed else parsed
                                    
                                    next_act = str(args.get("next_action", "route_to_planning")).lower()
                                    if next_act not in ["route_to_refinement", "route_to_planning", "finish"]:
                                        next_act = "route_to_planning"
                                        
                                    safe_args = {
                                        "reflection": args.get("reflection", args.get("反思", "Fallback")),
                                        "query_complexity": args.get("query_complexity", "complex"),
                                        "is_context_relevant": args.get("is_context_relevant", False) in [True, "true", "True", 1],
                                        "is_query_answered": args.get("is_query_answered", False) in [True, "true", "True", 1],
                                        "reasoning": args.get("reasoning", args.get("理由", "Fallback")),
                                        "feedback": args.get("feedback", args.get("反馈", "")),
                                        "next_action": next_act
                                    }
                                    decision_obj = ValidatorDecision(**safe_args)
                                    break
                            except Exception as e:
                                print(f"[VALIDATOR DEBUG] Errore parsing JSON manuale: {e}")
                                pass
                            i = j
                            break
                        if char == '\\': escape = not escape
                        else: escape = False
                i += 1
            
        if not decision_obj:
            print("[VALIDATOR] JSON non trovato. Uso fallback sicuro.")
            print(f"[VALIDATOR DEBUG] Raw LLM Response Content: {content_resp}")
            print(f"[VALIDATOR DEBUG] Raw LLM Tool Calls: {getattr(response_msg, 'tool_calls', 'NO TOOL CALLS')}")
            decision_obj = ValidatorDecision(reflection="Fallback", query_complexity="complex", is_context_relevant=False, is_query_answered=False, reasoning="Fallback", feedback="Fallback", next_action="route_to_planning")
            
        final_action = decision_obj.next_action
        if final_action == "finish" and num_plan == 0:
            final_action = "route_to_planning"
        elif final_action == "route_to_refinement" and num_ref >= 3:
            final_action = "route_to_planning"
        elif final_action == "route_to_planning" and num_plan >= 3:
            final_action = "finish"
            
        actions.append(final_action)
        if decision_obj.feedback:
            feedbacks.append(f"For sub-query '{q}': {decision_obj.feedback}")
            
    # Aggregate actions logically
    if "route_to_refinement" in actions:
        overall_action = "route_to_refinement"
    elif "route_to_planning" in actions:
        overall_action = "route_to_planning"
    else:
        overall_action = "finish"
        
    print(f"[VALIDATOR] Instradamento globale verso: {overall_action}")
    return {
        "next_node": overall_action,
        "feedback_history": feedbacks if feedbacks else []
    }
