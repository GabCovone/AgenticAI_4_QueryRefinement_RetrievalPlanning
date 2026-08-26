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
    num_ref = state.get("num_refinement", 0)
    num_plan = state.get("num_planning", 0)
    print(f"\n[VALIDATOR] Avvio ciclo decisionale (Ref={num_ref}, Plan={num_plan})...")
    
    llm_with_tools = llm.bind_tools([ValidatorDecision])
    
    system_prompt = """You are the Semantic Routing Validator.
Your job is to evaluate if the current queries are ready for retrieval or if the final answer satisfies the original user query.
If they are sub-queries, evaluate them individually.

CRITICAL LANGUAGE RULE: YOU MUST ONLY USE ENGLISH. Do not use or output any Chinese characters under any circumstances. You must strictly use the exact English keys for JSON: "reflection", "query_complexity", "is_context_relevant", "is_query_answered", "reasoning", "feedback", "next_action".
CRITICAL RULE 2: YOU WILL BE PENALIZED IF YOU RESOLVE ENTITIES USING INTERNAL KNOWLEDGE! If the query asks about an unknown entity (e.g., "the author of X", "the CEO of Y"), YOU MUST NOT write their real name in your feedback. You MUST strictly use the exact generic phrasing from the user's query or use placeholders like [Person]. DO NOT pre-answer the query in your feedback!

Options for 'next_action':
- 'route_to_refinement': if the sub-query is still too broad, ambiguous, or needs decomposition/expansion.
- 'route_to_planning': if the sub-query is well-formed and ready for document retrieval.
- 'finish': if the context already contains the definitive answer to the sub-query.

You MUST conclude your response with a valid JSON block calling the ValidatorDecision tool.

--- FEW-SHOT EXAMPLE (DO NOT COPY THIS! Use it ONLY as a structural reference) ---
{
  "name": "ValidatorDecision",
  "arguments": {
    "reflection": "The context provides the required information.",
    "query_complexity": "simple",
    "is_context_relevant": true,
    "is_query_answered": true,
    "reasoning": "The query is fully answered by the context.",
    "feedback": "None.",
    "next_action": "finish"
  }
}
"""

    query_raw = state.get("current_query", state.get("original_query", ""))
    queries = [q.strip() for q in query_raw.split("\n---\n") if q.strip()]
    if not queries:
        queries = [query_raw]
        
    context = state.get("retrieved_context", "")
    if state.get("final_answer"):
        context += f"\n\nFinal Answer: {state['final_answer']}"
        
    feedbacks = []
    actions = []
    
    for q_idx, q in enumerate(queries):
        print(f"[VALIDATOR] Valutazione sotto-query {q_idx+1}/{len(queries)}: '{q}'")
        try:
            response_msg = llm_with_tools.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Sub-Query: '{q}'.\nCurrent Context: '{context}'\nRefinement Count: {num_ref}\nPlanning Count: {num_plan}\n\nAnalyze the state and output the JSON tool call.")
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
            if t_call["name"] == "ValidatorDecision":
                args = t_call["args"]
                safe_args = {
                    "reflection": args.get("reflection", "Fallback"),
                    "query_complexity": args.get("query_complexity", "complex"),
                    "is_context_relevant": args.get("is_context_relevant", False) in [True, "true", "True", 1],
                    "is_query_answered": args.get("is_query_answered", False) in [True, "true", "True", 1],
                    "reasoning": args.get("reasoning", "Fallback"),
                    "feedback": args.get("feedback", ""),
                    "next_action": args.get("next_action", "route_to_planning")
                }
                decision_obj = ValidatorDecision(**safe_args)
        
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
                                parsed = json.loads(content_resp[i:j+1])
                                if "name" in parsed:
                                    args = parsed.get("arguments", parsed.get("args", parsed))
                                    safe_args = {
                                        "reflection": args.get("reflection", args.get("反思", "Fallback")),
                                        "query_complexity": args.get("query_complexity", "complex"),
                                        "is_context_relevant": args.get("is_context_relevant", False) in [True, "true", "True", 1],
                                        "is_query_answered": args.get("is_query_answered", False) in [True, "true", "True", 1],
                                        "reasoning": args.get("reasoning", args.get("理由", "Fallback")),
                                        "feedback": args.get("feedback", args.get("反馈", "")),
                                        "next_action": args.get("next_action", "route_to_planning")
                                    }
                                    decision_obj = ValidatorDecision(**safe_args)
                                    break
                            except Exception: pass
                            i = j
                            break
                        if char == '\\': escape = not escape
                        else: escape = False
                i += 1
            
        if not decision_obj:
            print("[VALIDATOR] JSON non trovato. Uso fallback sicuro.")
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
